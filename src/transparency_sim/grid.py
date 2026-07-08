"""Grid configuration utilities for Arm A collection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridCell:
    q: int
    c: float
    B: int
    series: tuple[str, ...]


@dataclass(frozen=True)
class GridConfig:
    config_name: str
    r: int
    depth: int | str
    delta: float
    cells: tuple[GridCell, ...]
    instances_per_cell: int
    reps_per_instance: int
    instance_seed_base: int
    live_allowed: bool
    corpus_seed_scope: str = "cell"


def load_config(path) -> GridConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    r = data["r"]
    depth = data["depth"]
    delta = data["delta"]
    if not isinstance(r, int) or r < 1:
        raise ValueError("r must be a positive integer")
    if depth != "inf" and not (isinstance(depth, int) and depth >= 1):
        raise ValueError("depth must be a positive integer or 'inf'")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if data["instances_per_cell"] < 1 or data["reps_per_instance"] < 1:
        raise ValueError("instances_per_cell and reps_per_instance must be positive")
    corpus_seed_scope = data.get("corpus_seed_scope", "cell")
    if corpus_seed_scope not in {"cell", "qc"}:
        raise ValueError("corpus_seed_scope must be 'cell' or 'qc'")

    merged: dict[tuple[int, float, int], set[str]] = {}
    for raw in data["cells"]:
        q = raw["q"]
        c = float(raw["c"])
        B = raw["B"]
        if not isinstance(q, int) or q < r:
            raise ValueError("cell q must be an integer with q >= r")
        if not (0.0 <= c <= 1.0):
            raise ValueError("cell c must lie in [0, 1]")
        if not isinstance(B, int) or B < 0:
            raise ValueError("cell B must be a non-negative integer")
        key = (q, c, B)
        merged.setdefault(key, set()).update(raw.get("series", ()))

    cells = tuple(
        GridCell(q=q, c=c, B=B, series=tuple(sorted(series)))
        for (q, c, B), series in merged.items()
    )
    return GridConfig(
        config_name=data["config_name"],
        r=r,
        depth=depth,
        delta=delta,
        cells=cells,
        instances_per_cell=data["instances_per_cell"],
        reps_per_instance=data["reps_per_instance"],
        instance_seed_base=data["instance_seed_base"],
        live_allowed=bool(data["live_allowed"]),
        corpus_seed_scope=corpus_seed_scope,
    )


def sorted_cells(config: GridConfig) -> tuple[GridCell, ...]:
    return tuple(sorted(config.cells, key=lambda cell: (cell.q, cell.c, cell.B)))


def qc_groups(config: GridConfig) -> tuple[tuple[int, float], ...]:
    groups = []
    seen = set()
    for cell in sorted_cells(config):
        key = (cell.q, cell.c)
        if key not in seen:
            seen.add(key)
            groups.append(key)
    return tuple(groups)


def qc_index(config: GridConfig, cell: GridCell) -> int:
    return qc_groups(config).index((cell.q, cell.c))


def corpus_seed(config: GridConfig, cell_index: int, instance_index: int) -> int:
    if config.corpus_seed_scope == "cell":
        return config.instance_seed_base + 1000 * cell_index + instance_index
    if config.corpus_seed_scope == "qc":
        cell = sorted_cells(config)[cell_index]
        return config.instance_seed_base + 1000 * qc_index(config, cell) + instance_index
    raise ValueError("corpus_seed_scope must be 'cell' or 'qc'")


def rep_seed(cell_index: int, instance_index: int, rep_index: int) -> int:
    return 100000 + 97 * cell_index + 13 * instance_index + rep_index


def run_key(cell: GridCell, instance_index: int, rep_index: int, arm: str) -> str:
    return f"q{cell.q}_c{cell.c}_B{cell.B}_i{instance_index}_r{rep_index}_{arm}"


def completed_keys(records_path) -> set[str]:
    path = Path(records_path)
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta = record.get("grid_meta")
        if isinstance(meta, dict) and isinstance(meta.get("run_key"), str):
            keys.add(meta["run_key"])
    return keys
