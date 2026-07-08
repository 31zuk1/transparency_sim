"""Validate Arm A JSONL records before promotion into data/."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.grid import load_config, run_key, sorted_cells  # noqa: E402


TOP_LEVEL_KEYS = {
    "schema_version", "instrument", "corpus", "budget", "depth",
    "conversation", "env_transcript", "answer_raw", "answer_scored",
    "distortion_answer", "distortion_recovery", "n_fetch_paid",
    "n_resolve", "n_protocol_errors", "n_sanitized_keys",
    "terminated_reason", "n_turns", "usage", "timestamp_utc", "grid_meta",
}
CORPUS_KEYS = {"q", "r", "c", "seed"}
GRID_META_KEYS = {
    "run_key", "arm", "config_name", "cell_index", "instance_index",
    "rep_index", "series",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--a0")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    records, schema_errors = read_and_validate_records(args.records)
    run_keys = [record["grid_meta"]["run_key"] for record in records if "grid_meta" in record]
    duplicates = sorted(key for key, count in Counter(run_keys).items() if count > 1)
    coverage = config_coverage(config, records)
    a0_missing = missing_a0_keys(records, args.a0) if args.a0 else []

    print(f"records: {len(records)} valid lines")
    print(f"schema_errors: {len(schema_errors)}")
    for error in schema_errors[:10]:
        print(f"schema error: {error}", file=sys.stderr)
    print(f"duplicate run_keys: {len(duplicates)}")
    for key in duplicates[:10]:
        print(f"duplicate run_key: {key}", file=sys.stderr)
    print(f"required run_keys: {coverage['required']}")
    print(f"present required run_keys: {coverage['present']}")
    print(f"missing run_keys: {len(coverage['missing'])}")
    for key in coverage["missing"][:10]:
        print(f"missing run_key: {key}")
    print(f"extra run_keys: {len(coverage['extra'])}")
    for key in coverage["extra"][:10]:
        print(f"warning: extra run_key: {key}", file=sys.stderr)
    if args.a0:
        print(f"a0_missing_keys: {len(a0_missing)}")
        for key in a0_missing[:10]:
            print(f"missing A0 key: {key}", file=sys.stderr)

    if schema_errors or duplicates or a0_missing:
        return 2
    return 0


def read_and_validate_records(records_path) -> tuple[list[dict], list[str]]:
    path = Path(records_path)
    records = []
    errors = []
    if not path.exists():
        return records, [f"{path} does not exist"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        record_errors = schema_errors(record)
        if record_errors:
            errors.extend(f"line {line_no}: {error}" for error in record_errors)
            continue
        records.append(record)
    return records, errors


def schema_errors(record: dict) -> list[str]:
    errors = []
    missing = sorted(TOP_LEVEL_KEYS - set(record))
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")
    corpus = record.get("corpus")
    if not isinstance(corpus, dict):
        errors.append("corpus must be an object")
    else:
        missing_corpus = sorted(CORPUS_KEYS - set(corpus))
        if missing_corpus:
            errors.append(f"missing corpus keys: {', '.join(missing_corpus)}")
    meta = record.get("grid_meta")
    if not isinstance(meta, dict):
        errors.append("grid_meta must be an object")
    else:
        missing_meta = sorted(GRID_META_KEYS - set(meta))
        if missing_meta:
            errors.append(f"missing grid_meta keys: {', '.join(missing_meta)}")
    for key in ("budget", "distortion_answer", "distortion_recovery"):
        if key not in record:
            continue
        try:
            float(record[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    return errors


def config_coverage(config, records: list[dict]) -> dict:
    arms = sorted({
        record["grid_meta"]["arm"] for record in records
        if record.get("grid_meta", {}).get("arm")
    }) or ["offline"]
    expected = set()
    for arm in arms:
        for cell in sorted_cells(config):
            for instance_index in range(config.instances_per_cell):
                for rep_index in range(config.reps_per_instance):
                    expected.add(run_key(cell, instance_index, rep_index, arm))
    present = {record["grid_meta"]["run_key"] for record in records}
    return {
        "required": len(expected),
        "present": len(present & expected),
        "missing": sorted(expected - present),
        "extra": sorted(present - expected),
    }


def missing_a0_keys(records: list[dict], a0_path) -> list[tuple]:
    path = Path(a0_path)
    needed = {
        (
            int(record["corpus"]["q"]),
            float(record["corpus"]["c"]),
            int(record["budget"]),
            int(record["corpus"]["seed"]),
        )
        for record in records
    }
    if not path.exists():
        return sorted(needed)
    with path.open(newline="", encoding="utf-8") as f:
        available = {
            (int(row["q"]), float(row["c"]), int(row["B"]), int(row["corpus_seed"]))
            for row in csv.DictReader(f)
        }
    return sorted(needed - available)


if __name__ == "__main__":
    raise SystemExit(main())
