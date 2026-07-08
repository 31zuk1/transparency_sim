"""Run Arm A grid arms: a0, offline rehearsal, or guarded live collection."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.a0 import a0_exact  # noqa: E402
from transparency_sim.generator import generate_corpus  # noqa: E402
from transparency_sim.grid import (  # noqa: E402
    completed_keys,
    corpus_seed,
    load_config,
    rep_seed,
    run_key,
    sorted_cells,
)
from transparency_sim.instrument import InstrumentSpec  # noqa: E402
from transparency_sim.llm_blind_id import run_llm_blind_id  # noqa: E402
from transparency_sim.llm_client import AnthropicClient, OpenAIClient, SequentialScriptClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=["a0", "offline", "live"], required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--max-new-runs", type=int, default=25)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.arm == "a0":
        return run_a0_arm(config)
    if args.arm == "offline":
        return run_offline_arm(config)
    return run_live_arm(config, args.provider, args.model, args.max_new_runs)


def run_a0_arm(config) -> int:
    out = ROOT / "outputs" / "results" / "a0_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = sorted_cells(config)
    for cell_index, group_cells in _a0_cell_groups(config, cells):
        q = group_cells[0].q
        c = group_cells[0].c
        b_values = _a0_b_values(group_cells)
        for instance_index in range(config.instances_per_cell):
            seed = corpus_seed(config, cell_index, instance_index)
            corpus = generate_corpus(q=q, r=config.r, c=c, seed=seed)
            for B in sorted(b_values):
                d1 = a0_exact(corpus, B, depth=1).distortion
                di = a0_exact(corpus, B, depth="inf").distortion
                rows.append({
                    "config_name": config.config_name,
                    "q": q,
                    "c": c,
                    "corpus_seed": seed,
                    "B": B,
                    "depth": config.depth,
                    "d_seed_1": d1,
                    "d_seed_inf": di,
                })
    _write_csv(out, rows, [
        "config_name", "q", "c", "corpus_seed", "B", "depth",
        "d_seed_1", "d_seed_inf",
    ])
    print(f"wrote {len(rows)} rows: {out}")
    return 0


def _a0_cell_groups(config, cells) -> list[tuple[int, tuple]]:
    if config.corpus_seed_scope == "cell":
        return [(cell_index, (cell,)) for cell_index, cell in enumerate(cells)]
    groups = []
    seen = set()
    for cell_index, cell in enumerate(cells):
        key = (cell.q, cell.c)
        if key in seen:
            continue
        seen.add(key)
        groups.append((cell_index, tuple(c for c in cells if (c.q, c.c) == key)))
    return groups


def _a0_b_values(cells) -> set[int]:
    q = cells[0].q
    b_values = {cell.B for cell in cells}
    if any("P3" in cell.series for cell in cells):
        step = max(1, q // 75)
        b_values.update(range(0, q + 1, step))
        b_values.add(q)
    return b_values


def run_offline_arm(config) -> int:
    record_path = ROOT / "outputs" / "runs" / "arm_a" / f"{config.config_name}_offline.jsonl"
    done = completed_keys(record_path)
    planned = _planned_runs(config, "offline")
    pending = [p for p in planned if p["run_key"] not in done]
    print(f"completed runs: {len(done)}")
    print(f"planned new runs: {len(pending)}")
    for plan in pending:
        instrument = InstrumentSpec(
            provider="offline",
            model="scripted-sequential",
            harness_version="1.1",
            max_turns=max(60, plan["cell"].B + config.r + 20),
            requested_seed=plan["rep_seed"],
        )
        corpus = generate_corpus(
            q=plan["cell"].q,
            r=config.r,
            c=plan["cell"].c,
            seed=plan["corpus_seed"],
        )
        result = run_llm_blind_id(
            corpus,
            SequentialScriptClient(policy_seed=plan["rep_seed"]),
            instrument,
            budget=plan["cell"].B,
            depth=config.depth,
            record_path=record_path,
            extra_meta=_extra_meta(config, plan, "offline"),
        )
        print(
            f"{plan['run_key']} D_hat={result.distortion_answer:.4f} "
            f"D_rec={result.distortion_recovery:.4f} "
            f"terminated={result.terminated_reason} paid={result.n_fetch_paid} "
            f"turns={result.n_turns}"
        )
    return 0


def run_live_arm(config, provider: str | None, model: str | None, max_new_runs: int) -> int:
    if not config.live_allowed:
        print("live arm refused: config.live_allowed is false", file=sys.stderr)
        return 2
    if provider not in {"anthropic", "openai"} or not model:
        print("--provider and --model are required for live", file=sys.stderr)
        return 2
    if max_new_runs > 200:
        print("--max-new-runs must be at most 200", file=sys.stderr)
        return 2
    key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key_name):
        print(f"{key_name} is not set", file=sys.stderr)
        return 2

    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    record_path = ROOT / "outputs" / "runs" / "arm_a" / (
        f"{config.config_name}_live_{provider}_{model_slug}.jsonl"
    )
    done = completed_keys(record_path)
    planned = _planned_runs(config, "live")
    pending = [p for p in planned if p["run_key"] not in done]
    print(f"instrument: {InstrumentSpec(provider=provider, model=model).to_dict()}")
    print(f"completed runs: {len(done)}")
    print(f"planned new runs: {len(pending)}")
    if len(pending) > max_new_runs:
        print("--max-new-runs cap exceeded; no live calls made", file=sys.stderr)
        return 2
    if not pending:
        return 0

    client_cls = AnthropicClient if provider == "anthropic" else OpenAIClient
    client = client_cls(model=model, temperature=0.0, max_output_tokens=300, allow_live=True)
    instrument = InstrumentSpec(provider=provider, model=model, temperature=0.0)
    for plan in pending:
        corpus = generate_corpus(
            q=plan["cell"].q,
            r=config.r,
            c=plan["cell"].c,
            seed=plan["corpus_seed"],
        )
        result = run_llm_blind_id(
            corpus,
            client,
            instrument,
            budget=plan["cell"].B,
            depth=config.depth,
            record_path=record_path,
            extra_meta=_extra_meta(config, plan, "live"),
        )
        print(
            f"{plan['run_key']} D_hat={result.distortion_answer:.4f} "
            f"D_rec={result.distortion_recovery:.4f} "
            f"terminated={result.terminated_reason} paid={result.n_fetch_paid} "
            f"turns={result.n_turns}"
        )
    return 0


def _planned_runs(config, arm: str) -> list[dict]:
    plans = []
    for cell_index, cell in enumerate(sorted_cells(config)):
        for instance_index in range(config.instances_per_cell):
            seed = corpus_seed(config, cell_index, instance_index)
            for rep_index in range(config.reps_per_instance):
                plans.append({
                    "cell": cell,
                    "cell_index": cell_index,
                    "instance_index": instance_index,
                    "rep_index": rep_index,
                    "corpus_seed": seed,
                    "rep_seed": rep_seed(cell_index, instance_index, rep_index),
                    "run_key": run_key(cell, instance_index, rep_index, arm),
                })
    return plans


def _extra_meta(config, plan: dict, arm: str) -> dict:
    return {
        "run_key": plan["run_key"],
        "arm": arm,
        "config_name": config.config_name,
        "cell_index": plan["cell_index"],
        "instance_index": plan["instance_index"],
        "rep_index": plan["rep_index"],
        "series": list(plan["cell"].series),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
