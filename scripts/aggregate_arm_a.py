"""Aggregate Arm A JSONL records into run and cell CSV tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.grid import load_config  # noqa: E402


RUN_FIELDS = [
    "run_key", "arm", "config_name", "series", "q", "c", "B", "corpus_seed",
    "rep_index", "d_hat", "d_rec", "gap", "budget_utilization", "n_resolve",
    "n_protocol_errors", "n_sanitized_keys", "terminated_reason", "n_turns",
    "input_tokens", "output_tokens", "harness_version",
]
CELL_FIELDS = [
    "q", "c", "B", "series", "n_runs", "mean_d_hat", "sd_d_hat",
    "mean_d_rec", "sd_d_rec", "mean_gap", "mean_budget_utilization",
    "protocol_error_rate", "n_terminated_answered",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--a0")
    parser.add_argument("--precision", nargs="*", type=float, default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    del config
    rows = read_run_rows(args.records)
    if args.a0:
        missing = missing_a0_keys(rows, args.a0)
        if missing:
            print(f"ERROR: A0 table is missing {len(missing)} record keys", file=sys.stderr)
            for key in missing[:10]:
                print(f"missing A0 key: {key}", file=sys.stderr)
            return 2
    out_dir = ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "arm_a_runs.csv", rows, RUN_FIELDS)
    cell_rows = cell_summary(rows)
    write_csv(out_dir / "arm_a_cells.csv", cell_rows, CELL_FIELDS)
    if args.precision:
        print_precision_table(cell_rows, args.precision)
    print(f"wrote {len(rows)} runs and {len(cell_rows)} cells")
    return 0


def read_run_rows(records_path) -> list[dict]:
    rows = []
    path = Path(records_path)
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        meta = record.get("grid_meta")
        if not isinstance(meta, dict):
            print(f"warning: skipping record without grid_meta at line {line_no}", file=sys.stderr)
            continue
        usage = record.get("usage") or {}
        corpus = record.get("corpus") or {}
        B = record["budget"]
        d_hat = record["distortion_answer"]
        d_rec = record["distortion_recovery"]
        rows.append({
            "run_key": meta["run_key"],
            "arm": meta["arm"],
            "config_name": meta["config_name"],
            "series": "|".join(meta.get("series", [])),
            "q": corpus.get("q"),
            "c": corpus.get("c"),
            "B": B,
            "corpus_seed": corpus.get("seed"),
            "rep_index": meta.get("rep_index"),
            "d_hat": d_hat,
            "d_rec": d_rec,
            "gap": d_hat - d_rec,
            "budget_utilization": record["n_fetch_paid"] / B if B else "",
            "n_resolve": record["n_resolve"],
            "n_protocol_errors": record["n_protocol_errors"],
            "n_sanitized_keys": record["n_sanitized_keys"],
            "terminated_reason": record["terminated_reason"],
            "n_turns": record["n_turns"],
            "input_tokens": usage.get("input_tokens", ""),
            "output_tokens": usage.get("output_tokens", ""),
            "harness_version": (record.get("instrument") or {}).get("harness_version", "1.0"),
        })
    return rows


def cell_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["q"], row["c"], row["B"], row["series"])].append(row)
    out = []
    for (q, c, B, series), group in sorted(grouped.items()):
        d_hat = [float(r["d_hat"]) for r in group]
        d_rec = [float(r["d_rec"]) for r in group]
        gaps = [float(r["gap"]) for r in group]
        util = [float(r["budget_utilization"]) for r in group if r["budget_utilization"] != ""]
        out.append({
            "q": q,
            "c": c,
            "B": B,
            "series": series,
            "n_runs": len(group),
            "mean_d_hat": statistics.fmean(d_hat),
            "sd_d_hat": statistics.stdev(d_hat) if len(d_hat) > 1 else "",
            "mean_d_rec": statistics.fmean(d_rec),
            "sd_d_rec": statistics.stdev(d_rec) if len(d_rec) > 1 else "",
            "mean_gap": statistics.fmean(gaps),
            "mean_budget_utilization": statistics.fmean(util) if util else "",
            "protocol_error_rate": sum(int(r["n_protocol_errors"]) > 0 for r in group) / len(group),
            "n_terminated_answered": sum(r["terminated_reason"] == "answered" for r in group),
        })
    return out


def n_required(sd: float, error: float) -> int:
    return math.ceil((1.96 * sd / error) ** 2)


def missing_a0_keys(run_rows: list[dict], a0_path) -> list[tuple]:
    path = Path(a0_path)
    if not path.exists():
        return sorted(set(_run_a0_key(row) for row in run_rows))
    with path.open(newline="", encoding="utf-8") as f:
        available = {
            (int(row["q"]), float(row["c"]), int(row["B"]), int(row["corpus_seed"]))
            for row in csv.DictReader(f)
        }
    needed = {_run_a0_key(row) for row in run_rows}
    return sorted(needed - available)


def _run_a0_key(row: dict) -> tuple:
    return (int(row["q"]), float(row["c"]), int(row["B"]), int(row["corpus_seed"]))


def print_precision_table(cell_rows: list[dict], errors: list[float]) -> None:
    print("precision table")
    for row in cell_rows:
        label = f"q={row['q']} c={row['c']} B={row['B']}"
        values = []
        for error in errors:
            if row["sd_d_hat"] == "":
                values.append(f"E={error}: insufficient")
            else:
                values.append(f"E={error}: {n_required(float(row['sd_d_hat']), error)}")
        print(label + " | " + " | ".join(values))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
