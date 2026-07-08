# Paired Grid and Contrast Estimator Work Report

Date: 2026-07-08
Repository: `31zuk1/transparency_sim`
Base commit: `172f817` (`Guard Arm A aggregation against A0 mismatches`)
Live API calls by Codex: none

## Scope

Implemented the paired-corpus grid mechanism and preregistered contrast
estimators requested in `NEXT_TASK_PAIRED_GRID_AND_CONTRASTS.md`.

This work keeps the default `corpus_seed_scope` as `"cell"` so existing grid
configs retain their previous seed derivation, A0 row counts, offline run
counts, and resume behavior.

## Changes

- Added `GridConfig.corpus_seed_scope` with allowed values `"cell"` and `"qc"`.
- Added `qc_groups(config)` and `qc_index(config, cell)`.
- Updated `corpus_seed(config, cell_index, instance_index)`:
  - `"cell"`: `instance_seed_base + 1000 * cell_index + instance_index`
  - `"qc"`: `instance_seed_base + 1000 * qc_index + instance_index`
- Added `configs/grid_rehearsal_paired.json` exactly as specified.
- Updated the A0 arm so `"qc"` scope generates each `(q, c, instance_index)`
  corpus once and evaluates the union of B values plus the P3 dense sweep.
- Added `scripts/analyze_contrasts.py`:
  - instance means are computed before contrast statistics;
  - P1/P2 use the specified one-sided Jonckheere-Terpstra statistic;
  - exact label permutations are used up to 100,000 assignments;
  - larger cases use seed `20260707` with 10,000 Monte Carlo assignments;
  - P3 reports B-threshold intervals, theory references, lower-bound audit
    flags, and c = 1 overshoot;
  - the same observed/recovery calculations are emitted with `_rec` columns;
  - A0 coverage is checked before output.
- Added `scripts/validate_records.py` for schema, duplicate run-key, config
  coverage, extra-key warning, and A0 coverage checks.
- Added the requested README section on paired grids and contrast estimators.
- Appended the eight specified tests to `tests/test_grid.py`.

## Validation

Commands run:

```bash
pytest
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm a0
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm offline
python -c "import csv; from pathlib import Path; print(sum(1 for _ in csv.DictReader(open('outputs/results/a0_grid.csv', encoding='utf-8')))); print(sum(1 for line in Path('outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl').open(encoding='utf-8') if line.strip()))"
python scripts/run_arm_a_grid.py --config configs/grid_rehearsal_paired.json --arm a0
python scripts/run_arm_a_grid.py --config configs/grid_rehearsal_paired.json --arm offline
python scripts/aggregate_arm_a.py --config configs/grid_rehearsal_paired.json --records outputs/runs/arm_a/arm_a_rehearsal_paired_offline.jsonl --a0 outputs/results/a0_grid.csv --precision 0.05 0.10
python scripts/analyze_contrasts.py --config configs/grid_rehearsal_paired.json --records outputs/runs/arm_a/arm_a_rehearsal_paired_offline.jsonl --a0 outputs/results/a0_grid.csv
python scripts/validate_records.py --records outputs/runs/arm_a/arm_a_rehearsal_paired_offline.jsonl --config configs/grid_rehearsal_paired.json --a0 outputs/results/a0_grid.csv
```

Results:

- `pytest`: 96 passed
- Pilot A0: 2,757 rows
- Pilot offline: 57 records; resume reported `planned new runs: 0`
- Paired A0: 64 rows and 4 unique `(q, c, corpus_seed)` corpora
- Paired offline: 6 records; resume reported `planned new runs: 0`
- Paired aggregate with A0 coverage: success
- Paired contrast analysis: wrote `outputs/results/arm_a_contrasts.csv`
- Paired record validation: 0 schema errors, 0 duplicate run keys, 0 missing
  required run keys, 0 extra run keys, 0 missing A0 keys

## Notes

No live calls were made. Generated `outputs/` artifacts remain local and
ignored by git. The estimator definitions were implemented as normative code
without changing prompts, protocols, the environment, scoring, or existing
estimators.
