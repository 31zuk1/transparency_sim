# Arm A Grid Work Report

Date: 2026-07-07
Repository: `31zuk1/transparency_sim`
Base commit: `8512cec` (`作業ログを追加`)
Live API calls by Codex: none

## Objective

Build the machinery for Arm A collection without running live APIs:

- exact A0 grid computation
- full offline rehearsal through the text protocol
- resumable live runner with guards
- run and cell aggregation
- Figure 2 generator
- operator runbook
- test coverage for grid behavior, resumption, aggregation, and figure output

## Implemented Files

New files:

- `src/transparency_sim/grid.py`
- `configs/grid_pilot.json`
- `configs/grid_rehearsal_offline.json`
- `scripts/run_arm_a_grid.py`
- `scripts/aggregate_arm_a.py`
- `scripts/make_fig2.py`
- `RUNBOOK_ARM_A.md`
- `data/arm_a/.gitkeep`
- `tests/test_grid.py`

Existing files changed:

- `src/transparency_sim/llm_client.py`
- `src/transparency_sim/instrument.py`
- `src/transparency_sim/llm_blind_id.py`
- `README.md`

No prompt text, observation format, protocol grammar, existing test files, or
environment/scoring/generator/A0/theory/plot modules were changed.

## Key Changes

### Transport hardening

`extract_openai_output_text(response)` now walks structured OpenAI Responses
API output parts and joins text parts with newlines. This protects the
final-line command parser from fused command strings observed in the live
pilot.

### Instrument versioning

`InstrumentSpec` now carries `harness_version="1.1"`. Older records without
that field remain readable and are treated by aggregation as `"1.0"`.

### Grid metadata

`run_llm_blind_id(..., extra_meta=...)` can now attach optional `grid_meta` to
JSONL records. Grid runs use this to store `run_key`, arm, config name, cell
index, instance index, repetition index, and series labels. Runs without
`extra_meta` keep the previous schema shape.

### Grid runner

`scripts/run_arm_a_grid.py` supports:

- `--arm a0`: exact `D_seed_1` and `D_seed_inf` computation, including dense
  P3 B sweeps
- `--arm offline`: full scripted-client rehearsal with resumable JSONL records
- `--arm live`: guarded live path only; requires live-enabled config, provider
  key, and `--max-new-runs` cap

Codex did not execute the live arm.

### Aggregation and Figure 2

`scripts/aggregate_arm_a.py` emits:

- `outputs/results/arm_a_runs.csv`
- `outputs/results/arm_a_cells.csv`
- precision table via `--precision`

`scripts/make_fig2.py` renders the Figure 2 layout. Offline sources receive
the mandatory watermark:

```text
OFFLINE REHEARSAL - scripted client - not a result
```

## Validation

Commands run successfully:

```bash
pytest
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm a0
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm offline
python scripts/aggregate_arm_a.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl --a0 outputs/results/a0_grid.csv --precision 0.05 0.10
python scripts/make_fig2.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl --a0 outputs/results/a0_grid.csv --source offline
python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm offline
python scripts/regenerate_fig1.py
python scripts/run_a0_smoke.py
python scripts/run_blind_id_dryrun.py
python scripts/run_llm_blind_id_dryrun.py
```

Observed checks:

- `pytest`: 86 passed
- A0 grid: 2,757 rows written to `outputs/results/a0_grid.csv`
- Offline grid rehearsal: 57 records written
- Offline resumption: second offline invocation reported `planned new runs: 0`
- Aggregation: 57 run rows and 19 cell rows
- Figure 2 rehearsal files generated under `outputs/figures/`
- Existing scripts still run

## Notes

Generated outputs remain under ignored `outputs/` paths. They are local
artifacts and are not committed. Finalized records can later be promoted into
`data/arm_a/` following `RUNBOOK_ARM_A.md`.

The live runner is implemented but was not executed. The next operator step is
to follow the runbook and collect live records in capped, resumable slices.
