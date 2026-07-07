"""LLM client abstractions for protocol-driven Blind-ID runs."""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Protocol

from .blind_id import extract_answers


@dataclass(frozen=True)
class LLMReply:
    text: str
    usage: dict | None


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict]) -> LLMReply:
        """messages are [{"role": "user"|"assistant", "content": str}, ...]."""


def extract_openai_output_text(response) -> str:
    """Join all text parts of a Responses API result with newlines.

    Some responses carry multiple output items; concatenating them without a
    separator can fuse commands (observed in the live pilot as e.g.
    'LISTLIST'). Joining with newlines lets the final-nonempty-line parse
    rule recover the last command. Falls back to `output_text` when the
    structured walk yields nothing.
    """
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    if parts:
        return "\n".join(parts)
    return getattr(response, "output_text", "") or ""


class AnthropicClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_output_tokens: int,
        allow_live: bool = False,
    ) -> None:
        if allow_live is not True:
            raise RuntimeError("live client requires allow_live=True")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install -r requirements-llm.txt") from exc
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def complete(self, system: str, messages: list[dict]) -> LLMReply:
        response = self._client.messages.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            system=system,
            messages=messages,
        )
        text = "\n".join(
            block.text for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        )
        usage_obj = getattr(response, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
            }
        return LLMReply(text=text, usage=usage)


class OpenAIClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_output_tokens: int,
        allow_live: bool = False,
    ) -> None:
        if allow_live is not True:
            raise RuntimeError("live client requires allow_live=True")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install -r requirements-llm.txt") from exc
        self._client = OpenAI(api_key=key)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def complete(self, system: str, messages: list[dict]) -> LLMReply:
        response = self._client.responses.create(
            model=self._model,
            instructions=system,
            input=messages,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        text = extract_openai_output_text(response)
        usage_obj = getattr(response, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
            }
        return LLMReply(text=text, usage=usage)


class TranscriptReplayClient:
    """Return prewritten replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    def complete(self, system: str, messages: list[dict]) -> LLMReply:
        if self._index >= len(self._replies):
            raise RuntimeError("transcript replay exhausted")
        reply = self._replies[self._index]
        self._index += 1
        return LLMReply(text=reply, usage=None)


class SequentialScriptClient:
    """Replay the scripted sequential policy through conversation text."""

    def __init__(self, policy_seed: int) -> None:
        self.policy_seed = policy_seed

    def complete(self, system: str, messages: list[dict]) -> LLMReply:
        state = _initial_state(messages[0]["content"])
        history = _history_pairs(messages)
        command = _next_script_command(state, history, self.policy_seed)
        return LLMReply(text=command, usage=None)


def _initial_state(text: str) -> dict:
    r_match = re.search(r"ANSWER SHEET \((\d+) components\):", text)
    b_match = re.search(r"BUDGET: (\d+) direct acquisitions", text)
    marker = "DOCUMENT IDS"
    if not (r_match and b_match and marker in text):
        raise RuntimeError("cannot parse initial task message")
    after_ids = text.split(marker, 1)[1]
    ids = tuple(re.findall(r"DOC_[0-9A-F]{4}", after_ids))
    return {"r": int(r_match.group(1)), "budget": int(b_match.group(1)), "ids": ids}


def _history_pairs(messages: list[dict]) -> list[tuple[str, dict]]:
    pairs: list[tuple[str, dict]] = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        command = _final_line(message.get("content", ""))
        observation = {}
        if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
            observation = _parse_observation(messages[i + 1].get("content", ""))
        pairs.append((command, observation))
    return pairs


def _final_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _parse_observation(text: str) -> dict:
    budget_match = re.search(r"BUDGET_REMAINING: (\d+)", text)
    budget = int(budget_match.group(1)) if budget_match else None
    if text.startswith("OK FETCH") or text.startswith("OK RESOLVE"):
        doc_match = re.search(r"^DOC (DOC_[0-9A-F]{4})$", text, re.MULTILINE)
        refs_match = re.search(r"^REFS: (.*)$", text, re.MULTILINE)
        body = text.split("BODY:\n", 1)[1] if "BODY:\n" in text else ""
        refs = ()
        if refs_match and refs_match.group(1) != "none":
            refs = tuple(re.findall(r"DOC_[0-9A-F]{4}", refs_match.group(1)))
        return {
            "ok": True,
            "kind": "fetch" if text.startswith("OK FETCH") else "resolve",
            "doc_id": doc_match.group(1) if doc_match else None,
            "refs": refs,
            "body": body,
            "budget": budget,
        }
    if text.startswith("ERROR "):
        name = text.split(":", 1)[0].replace("ERROR ", "")
        return {"ok": False, "error": name, "budget": budget}
    if text.startswith("PROTOCOL_ERROR"):
        return {"ok": False, "error": "PROTOCOL_ERROR", "budget": budget}
    if text.startswith("OK LIST"):
        return {"ok": True, "kind": "list", "budget": budget}
    return {"ok": False, "error": "UNKNOWN", "budget": budget}


def _next_script_command(state: dict, history: list[tuple[str, dict]], seed: int) -> str:
    pool = list(state["ids"])
    random.Random(seed).shuffle(pool)
    prior = list(history)
    bodies: dict[str, str] = {}
    refs_by_doc: dict[str, tuple[str, ...]] = {}
    obtained: set[str] = set()
    budget = state["budget"]

    def consume(expected: str):
        if not prior:
            return None
        command, observation = prior.pop(0)
        if command != expected:
            raise RuntimeError(f"unexpected replay command {command!r}, expected {expected!r}")
        if observation.get("budget") is not None:
            nonlocal_budget[0] = observation["budget"]
        return observation

    nonlocal_budget = [budget]
    for doc_id in pool:
        budget = nonlocal_budget[0]
        if budget == 0:
            return _answer_command(bodies.values(), state["r"])
        if doc_id in obtained:
            continue
        expected = f"FETCH {doc_id}"
        observation = consume(expected)
        if observation is None:
            return expected
        if not observation.get("ok"):
            continue
        obtained.add(doc_id)
        bodies[doc_id] = observation.get("body", "")
        refs_by_doc[doc_id] = observation.get("refs", ())
        frontier = [doc_id]
        while frontier:
            src_id = frontier.pop()
            for target_id in refs_by_doc.get(src_id, ()):
                if target_id in obtained:
                    continue
                expected = f"RESOLVE {src_id} {target_id}"
                observation = consume(expected)
                if observation is None:
                    return expected
                if not observation.get("ok"):
                    continue
                resolved_id = observation.get("doc_id") or target_id
                obtained.add(resolved_id)
                bodies[resolved_id] = observation.get("body", "")
                refs_by_doc[resolved_id] = observation.get("refs", ())
                frontier.append(resolved_id)
    return _answer_command(bodies.values(), state["r"])


def _answer_command(bodies, r: int) -> str:
    import json

    return "ANSWER " + json.dumps(extract_answers(bodies, r), sort_keys=True)
