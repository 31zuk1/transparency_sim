# Arm A A0 Coverage Guard Work Log

Date: 2026-07-07
Repository: `31zuk1/transparency_sim`
Base commit: `2a898b6` (`Add Arm A grid rehearsal pipeline`)
Live API calls by Codex: none

## Motivation

The previous Arm A pipeline could silently combine run records with an A0 table
from a different grid. That failure mode is plausible in operator workflows
because `outputs/` is ignored and several scripts can leave local rehearsal
artifacts behind.

The specific risk: `aggregate_arm_a.py` and `make_fig2.py` could accept records
whose `(q, c, B, corpus_seed)` keys were not covered by the supplied A0 table.
If that happened, summaries and figures might be produced from inconsistent
inputs without warning.

## Change

Added A0 coverage validation keyed by:

```text
(q, c, B, corpus_seed)
```

### Aggregation

When `scripts/aggregate_arm_a.py` is called with `--a0`, it now checks that
every record row has a matching A0 row. If any key is missing, the command
prints an error and exits nonzero before writing summary tables.

### Figure 2

`scripts/make_fig2.py` now checks that every plotted P3 observation has a
matching A0 row. If any key is missing, figure generation raises an error
instead of silently drawing from inconsistent inputs.

## Tests Added

Added two tests in `tests/test_grid.py`:

- `test_aggregate_refuses_missing_a0_coverage`
- `test_fig2_refuses_missing_a0_coverage`

These tests reproduce the operator-error class directly with mismatched A0
keys and verify that the pipeline refuses the input.

## Validation

Commands run:

```bash
pytest
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm a0
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm offline
python scripts/aggregate_arm_a.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl --a0 outputs/results/a0_grid.csv --precision 0.05 0.10
python scripts/make_fig2.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl --a0 outputs/results/a0_grid.csv --source offline
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm offline
```

Results:

- `pytest`: 88 passed
- A0 grid generation: success
- Offline grid rehearsal: 57 records
- Aggregation with matching A0 table: success
- Offline Figure 2 generation with matching A0 table: success
- Offline resumption: `planned new runs: 0`

## Notes

No live API calls were made. Generated outputs remain local under ignored
`outputs/` paths. This change is a guardrail for live collection and should be
kept before running the 57-run operator collection.
