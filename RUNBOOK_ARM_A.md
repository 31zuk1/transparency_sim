# RUNBOOK: Arm A live collection (operator steps; Codex never runs these)

0. Rehearse offline end to end (no keys):
   python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm a0
   python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm offline
   python scripts/aggregate_arm_a.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl --a0 outputs/results/a0_grid.csv
   python scripts/make_fig2.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_offline.jsonl --a0 outputs/results/a0_grid.csv --source offline

1. Live pilot collection (57 runs; resumable; run in slices):
   set -a; source .env; set +a
   python scripts/run_arm_a_grid.py --config configs/grid_pilot.json --arm live --provider openai --model <MODEL> --max-new-runs 25
   Repeat the same command until "planned new runs: 0".

2. Aggregate and inspect (protocol_error_rate, budget_utilization, gap):
   python scripts/aggregate_arm_a.py --config configs/grid_pilot.json --records outputs/runs/arm_a/arm_a_pilot_v1_live_openai_<MODEL>.jsonl --a0 outputs/results/a0_grid.csv --precision 0.05 0.10

3. Freeze the preregistration inputs (instance counts from the precision
   table), then promote the finalized records into the tracked data
   directory and commit:
   cp outputs/runs/arm_a/*.jsonl data/arm_a/
   git add data/arm_a && git commit -m "Arm A pilot records"

4. Do not generate the live figure 2 or touch the manuscript until the
   preregistered main grid is collected.
