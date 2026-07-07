"""Text-protocol Blind-ID harness for offline and live LLM clients."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from .corpus import Corpus
from .environment import BlindIDEnvironment, FetchView, TransparencySimEnvError
from .instrument import InstrumentSpec
from .llm_client import LLMClient
from .scoring import answer_sheet, recovery_distortion, score_answer


@dataclass(frozen=True)
class ListCmd:
    pass


@dataclass(frozen=True)
class FetchCmd:
    doc_id: str


@dataclass(frozen=True)
class ResolveCmd:
    src_id: str
    target_id: str


@dataclass(frozen=True)
class AnswerCmd:
    payload: dict


@dataclass(frozen=True)
class ProtocolViolation:
    message: str


Command = Union[ListCmd, FetchCmd, ResolveCmd, AnswerCmd]


SYSTEM_PROMPT_V1 = """You are an observer reconstructing facts about one policy decision from a document repository. You interact with the repository ONLY through commands. Before fetching, you see nothing but anonymous document ids.

Reply with exactly one command as the final line of your reply:
  LIST
  FETCH <doc_id>
  RESOLVE <src_id> <target_id>
  ANSWER <one-line JSON object>

Rules:
- FETCH acquires a document and costs 1 unit of your acquisition budget. Re-fetching a document you already fetched directly is free. Fetching a document you obtained only via RESOLVE costs 1.
- RESOLVE follows a reference id listed in a document you have obtained. It is free, but the chain depth is limited by the environment.
- Document ids look like DOC_XXXX. Never invent ids.
- When your budget is exhausted, or you are confident, reply with ANSWER. The JSON object maps answer-sheet keys to string values. Answer values must quote the exact phrase as written in the documents. Keys you omit are scored as incorrect."""

INITIAL_USER_TEMPLATE = """TASK: reconstruct the decision record and fill the answer sheet.
BUDGET: {budget} direct acquisitions. DEPTH LIMIT: {depth}.
ANSWER SHEET ({r} components):
{sheet_lines}
DOCUMENT IDS ({q}):
{ids}
Reply with your first command."""


@dataclass(frozen=True)
class LLMRunInfo:
    conversation: tuple[dict, ...]
    answer_raw: dict
    n_protocol_errors: int
    n_sanitized_keys: int
    terminated_reason: str
    n_turns: int
    usage: dict | None


def parse_command(text: str) -> Command | ProtocolViolation:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ProtocolViolation("empty reply")
    line = lines[-1]
    parts = line.split()
    if not parts:
        return ProtocolViolation("empty command")
    op = parts[0]
    if op == "LIST":
        if len(parts) != 1:
            return ProtocolViolation("LIST takes no arguments")
        return ListCmd()
    if op == "FETCH":
        if len(parts) != 2 or not _valid_doc_id(parts[1]):
            return ProtocolViolation("FETCH requires one document id")
        return FetchCmd(parts[1])
    if op == "RESOLVE":
        if len(parts) != 3 or not _valid_doc_id(parts[1]) or not _valid_doc_id(parts[2]):
            return ProtocolViolation("RESOLVE requires two document ids")
        return ResolveCmd(parts[1], parts[2])
    if op == "ANSWER":
        payload_text = line[len("ANSWER"):].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return ProtocolViolation("ANSWER payload is not valid JSON")
        if not isinstance(payload, dict):
            return ProtocolViolation("ANSWER payload must be an object")
        return AnswerCmd(payload)
    return ProtocolViolation("unknown command")


def sanitize_answer(payload: dict, valid_keys) -> tuple[dict[str, str], int]:
    valid = set(valid_keys)
    answer: dict[str, str] = {}
    dropped = 0
    for key, value in payload.items():
        if key not in valid or not isinstance(value, str):
            dropped += 1
            continue
        answer[key] = value
    return answer, dropped


class LLMBlindIDPolicy:
    """BlindIDPolicy implementation backed by a one-command text protocol."""

    def __init__(self, client: LLMClient, instrument: InstrumentSpec) -> None:
        self.client = client
        self.instrument = instrument
        self.last_run_info: LLMRunInfo | None = None

    def run(self, env: BlindIDEnvironment) -> dict[str, str]:
        ids = env.list_ids()
        sheet = answer_sheet(env.n_components)
        sheet_lines = "\n".join(f"{key}: {question}" for key, question in sheet)
        initial_user = INITIAL_USER_TEMPLATE.format(
            budget=env.budget_remaining,
            depth=_format_depth(self.instrument, env),
            r=env.n_components,
            q=len(ids),
            sheet_lines=sheet_lines,
            ids=" ".join(ids),
        )
        messages = [{"role": "user", "content": initial_user}]
        conversation = [{"role": "system", "content": SYSTEM_PROMPT_V1}, dict(messages[0])]
        valid_keys = {key for key, _ in sheet}
        n_errors = 0
        n_sanitized = 0
        usage = None
        n_turns = 0
        answer: dict[str, str] = {}
        answer_raw: dict = {}
        reason = "max_turns"

        while n_turns < self.instrument.max_turns:
            reply = self.client.complete(SYSTEM_PROMPT_V1, messages)
            n_turns += 1
            usage = _merge_usage(usage, reply.usage)
            messages.append({"role": "assistant", "content": reply.text})
            conversation.append({"role": "assistant", "content": reply.text})

            command = parse_command(reply.text)
            if isinstance(command, ProtocolViolation):
                n_errors += 1
                observation = _protocol_error_observation(
                    n_errors, self.instrument.max_protocol_errors
                )
                messages.append({"role": "user", "content": observation})
                conversation.append({"role": "user", "content": observation})
                if n_errors >= self.instrument.max_protocol_errors:
                    reason = "protocol_errors"
                    answer = {}
                    answer_raw = {}
                    break
                continue

            if isinstance(command, AnswerCmd):
                answer_raw = command.payload
                answer, n_sanitized = sanitize_answer(command.payload, valid_keys)
                reason = "answered"
                break

            observation = self._execute_command(env, command)
            messages.append({"role": "user", "content": observation})
            conversation.append({"role": "user", "content": observation})

        self.last_run_info = LLMRunInfo(
            conversation=tuple(conversation),
            answer_raw=answer_raw,
            n_protocol_errors=n_errors,
            n_sanitized_keys=n_sanitized,
            terminated_reason=reason,
            n_turns=n_turns,
            usage=usage,
        )
        return answer

    def _execute_command(self, env: BlindIDEnvironment, command: Command) -> str:
        try:
            if isinstance(command, ListCmd):
                ids = env.list_ids()
                return f"OK LIST\nIDS: {' '.join(ids)}\nBUDGET_REMAINING: {env.budget_remaining}"
            if isinstance(command, FetchCmd):
                view = env.fetch(command.doc_id)
                return _ok_document("FETCH", env.budget_remaining, view)
            if isinstance(command, ResolveCmd):
                view = env.resolve(command.src_id, command.target_id)
                return _ok_document("RESOLVE", env.budget_remaining, view)
        except TransparencySimEnvError as exc:
            text = (
                f"ERROR {exc.__class__.__name__}: {exc}\n"
                f"BUDGET_REMAINING: {env.budget_remaining}"
            )
            if exc.__class__.__name__ == "BudgetExhausted":
                text += "\nYou must reply with ANSWER now."
            return text
        raise AssertionError("unreachable command type")


@dataclass(frozen=True)
class LLMBlindIDRunResult:
    answer: dict[str, str]
    distortion_answer: float
    distortion_recovery: float
    n_fetch_paid: int
    n_resolve: int
    obtained: tuple[str, ...]
    n_protocol_errors: int
    n_sanitized_keys: int
    terminated_reason: str
    n_turns: int
    usage: dict | None


def run_llm_blind_id(
    corpus: Corpus,
    client: LLMClient,
    instrument: InstrumentSpec,
    budget: int,
    depth: int | str = "inf",
    record_path: Path | None = None,
) -> LLMBlindIDRunResult:
    env = BlindIDEnvironment(corpus=corpus, budget=budget, depth=depth, kappa=0.0)
    policy = LLMBlindIDPolicy(client=client, instrument=instrument)
    answer = policy.run(env)
    if policy.last_run_info is None:
        raise RuntimeError("policy did not produce run info")
    info = policy.last_run_info
    answer_score = score_answer(corpus.y0, answer)
    recovery = recovery_distortion(corpus, env.obtained_ids())
    transcript = env.transcript()
    result = LLMBlindIDRunResult(
        answer=answer,
        distortion_answer=answer_score.distortion,
        distortion_recovery=recovery,
        n_fetch_paid=sum(e.op == "fetch" and e.cost == 1 for e in transcript),
        n_resolve=sum(e.op == "resolve" for e in transcript),
        obtained=env.obtained_ids(),
        n_protocol_errors=info.n_protocol_errors,
        n_sanitized_keys=info.n_sanitized_keys,
        terminated_reason=info.terminated_reason,
        n_turns=info.n_turns,
        usage=info.usage,
    )
    if record_path is not None:
        _append_record(record_path, corpus, budget, depth, instrument, info, transcript, result)
    return result


def _valid_doc_id(value: str) -> bool:
    return value.startswith("DOC_")


def _format_depth(instrument: InstrumentSpec, env: BlindIDEnvironment) -> str:
    del instrument
    return "inf" if _depth_value(env) == "inf" else str(_depth_value(env))


def _depth_value(env: BlindIDEnvironment):
    return getattr(env, "_depth")


def _ok_document(op: str, budget_remaining: int, view: FetchView) -> str:
    refs = " ".join(view.refs) if view.refs else "none"
    return (
        f"OK {op}\n"
        f"BUDGET_REMAINING: {budget_remaining}\n"
        f"DOC {view.doc_id}\n"
        f"REFS: {refs}\n"
        f"BODY:\n{view.body}"
    )


def _protocol_error_observation(used: int, max_errors: int) -> str:
    return (
        "PROTOCOL_ERROR: reply with exactly one command as the final line. "
        f"({used}/{max_errors} errors used)"
    )


def _merge_usage(current: dict | None, new: dict | None) -> dict | None:
    if new is None:
        return current
    merged = dict(current or {})
    for key, value in new.items():
        if isinstance(value, (int, float)):
            merged[key] = merged.get(key, 0) + value
    return merged


def _append_record(
    record_path: Path,
    corpus: Corpus,
    budget: int,
    depth: int | str,
    instrument: InstrumentSpec,
    info: LLMRunInfo,
    transcript,
    result: LLMBlindIDRunResult,
) -> None:
    record = {
        "schema_version": 1,
        "instrument": instrument.to_dict(),
        "corpus": {"q": corpus.q, "r": corpus.r, "c": corpus.c, "seed": corpus.seed},
        "budget": budget,
        "depth": depth,
        "conversation": list(info.conversation),
        "env_transcript": [asdict(event) for event in transcript],
        "answer_raw": info.answer_raw,
        "answer_scored": result.answer,
        "distortion_answer": result.distortion_answer,
        "distortion_recovery": result.distortion_recovery,
        "n_fetch_paid": result.n_fetch_paid,
        "n_resolve": result.n_resolve,
        "n_protocol_errors": result.n_protocol_errors,
        "n_sanitized_keys": result.n_sanitized_keys,
        "terminated_reason": result.terminated_reason,
        "n_turns": result.n_turns,
        "usage": result.usage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
