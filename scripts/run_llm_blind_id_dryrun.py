"""Dry run for the protocol-driven Blind-ID LLM harness."""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.a0 import a0_exact  # noqa: E402
from transparency_sim.blind_id import ScriptedSequentialPolicy, run_blind_id  # noqa: E402
from transparency_sim.generator import generate_corpus  # noqa: E402
from transparency_sim.instrument import InstrumentSpec  # noqa: E402
from transparency_sim.llm_blind_id import run_llm_blind_id  # noqa: E402
from transparency_sim.llm_client import AnthropicClient, OpenAIClient, SequentialScriptClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args(argv)
    if args.live:
        return _run_live(args)
    return _run_offline()


def _run_offline() -> int:
    q, r, c, seed = 50, 5, 0.5, 2
    budget, depth = 10, "inf"
    corpus = generate_corpus(q=q, r=r, c=c, seed=seed)
    record_path = ROOT / "outputs" / "runs" / "llm_blind_id" / "offline_dryrun.jsonl"
    log_path = ROOT / "outputs" / "logs" / "llm_blind_id_dryrun.txt"
    if record_path.exists():
        record_path.unlink()

    lines = [
        "LLM Blind-ID dry run -- OFFLINE transport-fidelity check (no LLM, no network)",
        (
            "corpus: q=50 r=5 c=0.5 seed=2 | observer: B=10 depth=inf | "
            "instrument: offline/scripted-sequential"
        ),
        f"A0 reference on the same corpus: D_seed_inf = {a0_exact(corpus, budget, depth).distortion:.4f}",
        "",
        "seed  D_hat(protocol)  D_hat(direct)  obtained_set_equal  terminated",
    ]
    all_equal = True
    all_answered = True
    for s in range(10):
        instrument = InstrumentSpec(
            provider="offline",
            model="scripted-sequential",
            requested_seed=s,
        )
        protocol = run_llm_blind_id(
            corpus,
            SequentialScriptClient(policy_seed=s),
            instrument,
            budget=budget,
            depth=depth,
            record_path=record_path,
        )
        direct = run_blind_id(
            corpus,
            ScriptedSequentialPolicy(policy_seed=s),
            budget=budget,
            depth=depth,
        )
        same = (
            abs(protocol.distortion_answer - direct.distortion_answer) <= 1e-12
            and set(protocol.obtained) == set(direct.obtained)
        )
        all_equal = all_equal and same
        all_answered = all_answered and protocol.terminated_reason == "answered"
        lines.append(
            f"{s:>4}  {protocol.distortion_answer:>16.4f}  "
            f"{direct.distortion_answer:>13.4f}  {'OK' if same else 'FAIL':>18}  "
            f"{protocol.terminated_reason:>10}"
        )
    lines.extend(
        [
            "",
            (
                "invariants: protocol == direct in all 10 runs: "
                f"{'OK' if all_equal else 'FAIL'} | all terminated by ANSWER: "
                f"{'OK' if all_answered else 'FAIL'}"
            ),
            "records: outputs/runs/llm_blind_id/offline_dryrun.jsonl",
            "log written: outputs/logs/llm_blind_id_dryrun.txt",
        ]
    )
    text = "\n".join(lines)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if all_equal and all_answered else 1


def _run_live(args) -> int:
    runs = 2 if args.runs is None else args.runs
    if runs > 5:
        print("--runs must be at most 5", file=sys.stderr)
        return 2
    if not args.model:
        print("--model is required with --live", file=sys.stderr)
        return 2
    key_name = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key_name):
        print(f"{key_name} is not set", file=sys.stderr)
        return 2

    q, r, c, seed = 50, 5, 0.5, 2
    budget, depth = 10, "inf"
    corpus = generate_corpus(q=q, r=r, c=c, seed=seed)
    instrument = InstrumentSpec(provider=args.provider, model=args.model, temperature=0.0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    record_path = (
        ROOT / "outputs" / "runs" / "llm_blind_id"
        / f"live_{args.provider}_{model_slug}_{stamp}.jsonl"
    )
    print(f"instrument: {instrument.to_dict()}")
    print(f"max calls: {runs * instrument.max_turns}")
    client_cls = AnthropicClient if args.provider == "anthropic" else OpenAIClient
    client = client_cls(
        model=args.model,
        temperature=instrument.temperature,
        max_output_tokens=instrument.max_output_tokens,
        allow_live=True,
    )
    results = []
    for _ in range(runs):
        results.append(
            run_llm_blind_id(
                corpus,
                client,
                instrument,
                budget=budget,
                depth=depth,
                record_path=record_path,
            )
        )
    mean_d = sum(r.distortion_answer for r in results) / len(results)
    reasons = Counter(r.terminated_reason for r in results)
    print(f"records: {record_path}")
    print(f"mean D_hat = {mean_d:.4f}")
    print(f"A0 reference D_seed_inf = {a0_exact(corpus, budget, depth).distortion:.4f}")
    print(f"terminated: {dict(reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
