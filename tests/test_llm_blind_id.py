import json
import re
from pathlib import Path

import pytest

from transparency_sim.blind_id import ScriptedSequentialPolicy, run_blind_id
from transparency_sim.generator import generate_corpus
from transparency_sim.instrument import InstrumentSpec
from transparency_sim.llm_blind_id import LLMBlindIDPolicy, run_llm_blind_id
from transparency_sim.llm_client import (
    AnthropicClient,
    OpenAIClient,
    SequentialScriptClient,
    TranscriptReplayClient,
)


def test_offline_equivalence_with_scripted_policy():
    corpus = generate_corpus(q=50, r=5, c=0.5, seed=2)
    for seed in range(5):
        instrument = InstrumentSpec(
            provider="offline",
            model="scripted-sequential",
            requested_seed=seed,
        )
        protocol = run_llm_blind_id(
            corpus,
            SequentialScriptClient(policy_seed=seed),
            instrument,
            budget=10,
            depth="inf",
        )
        direct = run_blind_id(
            corpus,
            ScriptedSequentialPolicy(policy_seed=seed),
            budget=10,
            depth="inf",
        )
        assert protocol.distortion_answer == pytest.approx(direct.distortion_answer)
        assert set(protocol.obtained) == set(direct.obtained)


def test_initial_prompt_is_blind():
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=61)
    env_run = _run_policy_with_replies(corpus, ["ANSWER {}"], budget=10)
    info = env_run.last_run_info
    assert info is not None
    prompt_text = "\n".join(m["content"] for m in info.conversation[:2])

    for value in corpus.y0.values():
        assert value not in prompt_text
    assert "relevant document" not in prompt_text.lower()
    assert re.search(r"\bcore\b", prompt_text.lower()) is None


def test_budget_exhaustion_forces_answer_message():
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=62)
    first, second = corpus.docs[0].doc_id, corpus.docs[1].doc_id
    policy = _run_policy_with_replies(
        corpus,
        [f"FETCH {first}", f"FETCH {second}", "ANSWER {}"],
        budget=1,
    )
    info = policy.last_run_info
    assert info is not None
    text = "\n".join(m["content"] for m in info.conversation)

    assert "You must reply with ANSWER now." in text
    assert info.terminated_reason == "answered"


def test_protocol_error_limit_returns_empty_answer():
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=63)
    instrument = InstrumentSpec(
        provider="offline",
        model="replay",
        max_protocol_errors=2,
    )
    result = run_llm_blind_id(
        corpus,
        TranscriptReplayClient(["nonsense", "still wrong"]),
        instrument,
        budget=10,
    )

    assert result.answer == {}
    assert result.terminated_reason == "protocol_errors"
    assert result.distortion_answer == 1.0


def test_max_turns_terminates():
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=64)
    instrument = InstrumentSpec(provider="offline", model="replay", max_turns=2)
    result = run_llm_blind_id(
        corpus,
        TranscriptReplayClient(["LIST", "LIST"]),
        instrument,
        budget=10,
    )

    assert result.terminated_reason == "max_turns"


def test_environment_errors_are_relayed_not_raised():
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=65)
    policy = _run_policy_with_replies(
        corpus,
        ["FETCH DOC_NOTREAL", "RESOLVE DOC_NOTREAL DOC_ALSOFAKE", "ANSWER {}"],
        budget=10,
    )
    info = policy.last_run_info
    assert info is not None
    text = "\n".join(m["content"] for m in info.conversation)

    assert "ERROR UnknownDocument" in text
    assert "ERROR InvalidResolve" in text


def test_live_clients_require_opt_in_before_key_check():
    with pytest.raises(RuntimeError, match="allow_live"):
        AnthropicClient("model", 0.0, 10, allow_live=False)
    with pytest.raises(RuntimeError, match="allow_live"):
        OpenAIClient("model", 0.0, 10, allow_live=False)


def test_replay_client_exhaustion_raises():
    client = TranscriptReplayClient([])

    with pytest.raises(RuntimeError):
        client.complete("", [])


def test_run_record_schema_and_no_secrets(tmp_path, monkeypatch):
    fake_secret = "sk" + "-test-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", fake_secret)
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=66)
    path = tmp_path / "runs.jsonl"
    instrument = InstrumentSpec(provider="offline", model="replay")
    run_llm_blind_id(
        corpus,
        TranscriptReplayClient(["ANSWER {}"]),
        instrument,
        budget=10,
        record_path=path,
    )

    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    required = {
        "schema_version", "instrument", "corpus", "budget", "depth",
        "conversation", "env_transcript", "answer_raw", "answer_scored",
        "distortion_answer", "distortion_recovery", "n_fetch_paid", "n_resolve",
        "n_protocol_errors", "n_sanitized_keys", "terminated_reason", "n_turns",
        "usage", "timestamp_utc",
    }
    assert required <= set(record)
    assert fake_secret not in text
    assert ("sk" + "-") not in text


def test_records_are_appended_jsonl(tmp_path):
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=67)
    path = tmp_path / "runs.jsonl"
    instrument = InstrumentSpec(provider="offline", model="replay")

    run_llm_blind_id(corpus, TranscriptReplayClient(["ANSWER {}"]), instrument, 10, record_path=path)
    run_llm_blind_id(corpus, TranscriptReplayClient(["ANSWER {}"]), instrument, 10, record_path=path)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert all(json.loads(row)["schema_version"] == 1 for row in rows)


def test_offline_dryrun_exits_zero():
    from scripts.run_llm_blind_id_dryrun import main

    assert main([]) == 0
    assert Path("outputs/logs/llm_blind_id_dryrun.txt").exists()
    assert Path("outputs/runs/llm_blind_id/offline_dryrun.jsonl").exists()


def test_instrument_spec_roundtrip_and_validation():
    spec = InstrumentSpec(provider="offline", model="scripted-sequential", requested_seed=3)

    assert InstrumentSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        InstrumentSpec(provider="bad", model="x")
    with pytest.raises(ValueError):
        InstrumentSpec(provider="offline", model="x", max_turns=0)


def _run_policy_with_replies(corpus, replies, budget):
    from transparency_sim.environment import BlindIDEnvironment

    policy = LLMBlindIDPolicy(
        TranscriptReplayClient(replies),
        InstrumentSpec(provider="offline", model="replay"),
    )
    env = BlindIDEnvironment(corpus, budget=budget)
    policy.run(env)
    return policy
