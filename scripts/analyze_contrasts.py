"""Analyze preregistered Arm A contrasts from grid JSONL records."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.grid import load_config, sorted_cells  # noqa: E402
from transparency_sim.theory import (  # noqa: E402
    budget_c0_linear,
    budget_c1_upper,
    budget_lower,
)


BASE_FIELDS = [
    "contrast_id", "series", "c", "n_instances_per_group", "delta_hat",
    "jt_U", "p_exact", "n_perms", "exact_or_mc", "bstar_lo", "bstar_hi",
    "bstar_a0_lo", "bstar_a0_hi", "theory_lower", "theory_c1_upper",
    "theory_c0_linear", "lower_violation", "overshoot",
]
REC_FIELDS = [
    "delta_hat_rec", "jt_U_rec", "p_exact_rec", "n_perms_rec",
    "exact_or_mc_rec", "bstar_lo_rec", "bstar_hi_rec",
    "bstar_a0_lo_rec", "bstar_a0_hi_rec", "lower_violation_rec",
    "overshoot_rec",
]
FIELDS = BASE_FIELDS + REC_FIELDS
MC_SEED = 20260707
MC_N = 10000
EXACT_LIMIT = 100000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--a0", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    rows = read_record_rows(args.records)
    missing = missing_a0_keys(rows, args.a0)
    if missing:
        print(f"ERROR: A0 table is missing {len(missing)} record keys", file=sys.stderr)
        for key in missing[:10]:
            print(f"missing A0 key: {key}", file=sys.stderr)
        return 2

    a0_rows = read_a0_rows(args.a0)
    contrast_rows = analyze_contrasts(config, rows, a0_rows)
    out = ROOT / "outputs" / "results" / "arm_a_contrasts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, contrast_rows)
    print_table(contrast_rows)
    print(f"wrote {len(contrast_rows)} contrasts: {out}")
    return 0


def read_record_rows(records_path) -> list[dict]:
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
            raise ValueError(f"record line {line_no} is missing grid_meta")
        corpus = record.get("corpus") or {}
        rows.append({
            "run_key": meta["run_key"],
            "q": int(corpus["q"]),
            "c": float(corpus["c"]),
            "B": int(record["budget"]),
            "corpus_seed": int(corpus["seed"]),
            "instance_index": int(meta["instance_index"]),
            "rep_index": int(meta["rep_index"]),
            "d_hat": float(record["distortion_answer"]),
            "d_rec": float(record["distortion_recovery"]),
        })
    return rows


def read_a0_rows(a0_path) -> list[dict]:
    with Path(a0_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze_contrasts(config, run_rows: list[dict], a0_rows: list[dict]) -> list[dict]:
    means = instance_means(run_rows)
    rows = []
    rows.extend(_ordered_series_rows(config, means, "P1"))
    rows.extend(_ordered_series_rows(config, means, "P2"))
    rows.extend(_p3_rows(config, means, a0_rows))
    return rows


def instance_means(run_rows: list[dict]) -> dict[tuple[int, float, int], list[dict]]:
    grouped: dict[tuple[int, float, int, int], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["q"], row["c"], row["B"], row["instance_index"])].append(row)

    by_cell: dict[tuple[int, float, int], list[dict]] = defaultdict(list)
    for (q, c, B, instance_index), group in sorted(grouped.items()):
        by_cell[(q, c, B)].append({
            "instance_index": instance_index,
            "corpus_seed": group[0]["corpus_seed"],
            "d_hat": statistics.fmean(row["d_hat"] for row in group),
            "d_rec": statistics.fmean(row["d_rec"] for row in group),
        })
    for key in by_cell:
        by_cell[key].sort(key=lambda row: row["instance_index"])
    return dict(by_cell)


def _ordered_series_rows(config, means: dict, series: str) -> list[dict]:
    cells = [cell for cell in sorted_cells(config) if series in cell.series]
    if series == "P1":
        rows = []
        for c in sorted({cell.c for cell in cells}):
            c_cells = [cell for cell in cells if cell.c == c]
            b_values = {cell.B for cell in c_cells}
            if len(c_cells) != 3 or len(b_values) != 1:
                continue
            rows.append(_ordered_row(f"P1_c{c}", "P1", c, c_cells, means))
        return rows
    if len(cells) != 3:
        return []
    return [_ordered_row("P2_path", "P2", "", cells, means)]


def _ordered_row(contrast_id: str, series: str, c, cells: list, means: dict) -> dict:
    cells = sorted(cells, key=lambda cell: cell.q)
    hat_groups = [_cell_values(means, cell, "d_hat") for cell in cells]
    rec_groups = [_cell_values(means, cell, "d_rec") for cell in cells]
    hat = ordered_contrast_stats(hat_groups)
    rec = ordered_contrast_stats(rec_groups)
    row = _blank_row()
    row.update({
        "contrast_id": contrast_id,
        "series": series,
        "c": c,
        "n_instances_per_group": _n_instances_label([len(group) for group in hat_groups]),
        "delta_hat": hat["delta_hat"],
        "jt_U": hat["jt_U"],
        "p_exact": hat["p_exact"],
        "n_perms": hat["n_perms"],
        "exact_or_mc": hat["exact_or_mc"],
        "delta_hat_rec": rec["delta_hat"],
        "jt_U_rec": rec["jt_U"],
        "p_exact_rec": rec["p_exact"],
        "n_perms_rec": rec["n_perms"],
        "exact_or_mc_rec": rec["exact_or_mc"],
    })
    return row


def ordered_contrast_stats(groups: list[list[float]]) -> dict:
    if not groups or any(not group for group in groups):
        raise ValueError("ordered contrast groups must be non-empty")
    u_obs = jt_u(groups)
    p_value, n_perms, exact_or_mc = jt_p_value(groups, u_obs)
    return {
        "delta_hat": statistics.fmean(groups[-1]) - statistics.fmean(groups[0]),
        "jt_U": u_obs,
        "p_exact": p_value,
        "n_perms": n_perms,
        "exact_or_mc": exact_or_mc,
    }


def jt_u(groups: list[list[float]]) -> float:
    total = 0.0
    for i, lower in enumerate(groups[:-1]):
        for higher in groups[i + 1:]:
            for x in lower:
                for y in higher:
                    if y > x:
                        total += 1.0
                    elif y == x:
                        total += 0.5
    return total


def jt_p_value(groups: list[list[float]], u_obs: float | None = None) -> tuple[float, int, str]:
    if u_obs is None:
        u_obs = jt_u(groups)
    values = [value for group in groups for value in group]
    sizes = [len(group) for group in groups]
    n_perms = _multinomial_count(sizes)
    if n_perms <= EXACT_LIMIT:
        ge = 0
        for permuted in _group_assignments(values, sizes):
            if jt_u(permuted) >= u_obs:
                ge += 1
        return ge / n_perms, n_perms, "exact"

    rng = random.Random(MC_SEED)
    ge = 0
    for _ in range(MC_N):
        shuffled = list(values)
        rng.shuffle(shuffled)
        permuted = []
        offset = 0
        for size in sizes:
            permuted.append(shuffled[offset:offset + size])
            offset += size
        if jt_u(permuted) >= u_obs:
            ge += 1
    return (ge + 1) / (MC_N + 1), MC_N, "mc"


def _multinomial_count(sizes: list[int]) -> int:
    remaining = sum(sizes)
    total = 1
    for size in sizes:
        total *= math.comb(remaining, size)
        remaining -= size
    return total


def _group_assignments(values: list[float], sizes: list[int]):
    indices = tuple(range(len(values)))

    def build(remaining: tuple[int, ...], group_index: int):
        if group_index == len(sizes) - 1:
            if len(remaining) == sizes[group_index]:
                yield [[values[i] for i in remaining]]
            return
        for combo in combinations(remaining, sizes[group_index]):
            combo_set = set(combo)
            next_remaining = tuple(i for i in remaining if i not in combo_set)
            for tail in build(next_remaining, group_index + 1):
                yield [[values[i] for i in combo]] + tail

    yield from build(indices, 0)


def _p3_rows(config, means: dict, a0_rows: list[dict]) -> list[dict]:
    p3_cells = [cell for cell in sorted_cells(config) if "P3" in cell.series]
    rows = []
    for q in sorted({cell.q for cell in p3_cells}):
        for c in sorted({cell.c for cell in p3_cells if cell.q == q}):
            cells = [cell for cell in p3_cells if cell.q == q and cell.c == c]
            rows.append(_p3_row(config, q, c, cells, means, a0_rows))
    return rows


def _p3_row(config, q: int, c: float, cells: list, means: dict, a0_rows: list[dict]) -> dict:
    hat_points = _p3_points(cells, means, "d_hat")
    rec_points = _p3_points(cells, means, "d_rec")
    a0_points = _a0_points(a0_rows, q, c)
    theory_lower = budget_lower(q, config.r, config.delta)
    theory_c1_upper = budget_c1_upper(q, config.r, config.delta)
    theory_c0_linear = budget_c0_linear(q, config.delta)
    hat = p3_stats(hat_points, config.delta, theory_lower, theory_c1_upper, c)
    rec = p3_stats(rec_points, config.delta, theory_lower, theory_c1_upper, c)
    a0_lo, a0_hi = bstar_interval(a0_points, config.delta)

    row = _blank_row()
    row.update({
        "contrast_id": f"P3_q{q}_c{c}",
        "series": "P3",
        "c": c,
        "n_instances_per_group": _n_instances_label([
            len(means.get((cell.q, cell.c, cell.B), [])) for cell in sorted(cells, key=lambda cell: cell.B)
        ]),
        "bstar_lo": hat["bstar_lo"],
        "bstar_hi": hat["bstar_hi"],
        "bstar_a0_lo": a0_lo,
        "bstar_a0_hi": a0_hi,
        "theory_lower": theory_lower,
        "theory_c1_upper": theory_c1_upper,
        "theory_c0_linear": theory_c0_linear,
        "lower_violation": hat["lower_violation"],
        "overshoot": hat["overshoot"],
        "bstar_lo_rec": rec["bstar_lo"],
        "bstar_hi_rec": rec["bstar_hi"],
        "bstar_a0_lo_rec": a0_lo,
        "bstar_a0_hi_rec": a0_hi,
        "lower_violation_rec": rec["lower_violation"],
        "overshoot_rec": rec["overshoot"],
    })
    return row


def _p3_points(cells: list, means: dict, measure: str) -> list[tuple[int, float]]:
    points = []
    for cell in sorted(cells, key=lambda cell: cell.B):
        values = _cell_values(means, cell, measure)
        if values:
            points.append((cell.B, statistics.fmean(values)))
    return points


def _a0_points(a0_rows: list[dict], q: int, c: float) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in a0_rows:
        if int(row["q"]) == q and float(row["c"]) == c:
            grouped[int(row["B"])].append(float(row["d_seed_inf"]))
    return [(B, statistics.fmean(values)) for B, values in sorted(grouped.items())]


def p3_stats(points: list[tuple[int, float]], delta: float, theory_lower: float,
             theory_c1_upper: float, c: float) -> dict:
    lo, hi = bstar_interval(points, delta)
    lower_violation = any(B < theory_lower and mean <= delta for B, mean in points)
    overshoot = ""
    if c == 1.0:
        overshoot = "not reached" if hi is None else max(0.0, hi - theory_c1_upper)
    return {
        "bstar_lo": lo,
        "bstar_hi": hi,
        "lower_violation": lower_violation,
        "overshoot": overshoot,
    }


def bstar_interval(points: list[tuple[int, float]], delta: float) -> tuple[int, int | None]:
    sorted_points = sorted(points)
    hi_values = [B for B, mean in sorted_points if mean <= delta]
    lo_values = [B for B, mean in sorted_points if mean > delta]
    return (max(lo_values) if lo_values else 0, min(hi_values) if hi_values else None)


def missing_a0_keys(run_rows: list[dict], a0_path) -> list[tuple]:
    path = Path(a0_path)
    if not path.exists():
        return sorted({_run_a0_key(row) for row in run_rows})
    with path.open(newline="", encoding="utf-8") as f:
        available = {
            (int(row["q"]), float(row["c"]), int(row["B"]), int(row["corpus_seed"]))
            for row in csv.DictReader(f)
        }
    needed = {_run_a0_key(row) for row in run_rows}
    return sorted(needed - available)


def _run_a0_key(row: dict) -> tuple:
    return (int(row["q"]), float(row["c"]), int(row["B"]), int(row["corpus_seed"]))


def _cell_values(means: dict, cell, measure: str) -> list[float]:
    return [row[measure] for row in means.get((cell.q, cell.c, cell.B), [])]


def _n_instances_label(sizes: list[int]) -> str:
    if not sizes:
        return ""
    if all(size == sizes[0] for size in sizes):
        return str(sizes[0])
    return "|".join(str(size) for size in sizes)


def _blank_row() -> dict:
    return {field: "" for field in FIELDS}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("contrast_id series c")
        print("(no contrasts)")
        return
    print("contrast_id series c delta_hat p_exact bstar_lo bstar_hi lower_violation")
    for row in rows:
        print(
            f"{row['contrast_id']} {row['series']} {row['c']} "
            f"{row['delta_hat']} {row['p_exact']} {row['bstar_lo']} "
            f"{row['bstar_hi']} {row['lower_violation']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
