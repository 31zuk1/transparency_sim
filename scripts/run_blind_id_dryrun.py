"""Dry run for the scripted Blind-ID arm."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.a0 import a0_exact  # noqa: E402
from transparency_sim.blind_id import NullPolicy, ScriptedSequentialPolicy, run_blind_id  # noqa: E402
from transparency_sim.generator import generate_corpus  # noqa: E402


def main() -> int:
    q, r, c, seed = 50, 5, 0.5, 2
    budget, depth = 10, "inf"
    corpus = generate_corpus(q=q, r=r, c=c, seed=seed)
    a0_d1 = a0_exact(corpus, budget, depth=1)
    a0_di = a0_exact(corpus, budget, depth=depth)

    runs = [
        run_blind_id(corpus, ScriptedSequentialPolicy(policy_seed=s), budget=budget, depth=depth)
        for s in range(20)
    ]
    null_run = run_blind_id(corpus, NullPolicy(), budget=budget, depth=depth)

    lines = [
        "Blind-ID dry run (scripted sequential policy; no LLM in this round)",
        "corpus: q=50 r=5 c=0.5 seed=2 | observer: B=10 depth=inf kappa=0",
        (
            "A0 reference on the same corpus: "
            f"D_seed_1 = {a0_d1.distortion:.4f}  D_seed_inf = {a0_di.distortion:.4f}"
        ),
        "",
        "seed  paid_fetch  resolves  recovered/r  D_hat  D_rec  match",
    ]

    all_match = True
    all_paid = True
    for s, run in enumerate(runs):
        match = abs(run.distortion_answer - run.distortion_recovery) <= 1e-12
        all_match = all_match and match
        all_paid = all_paid and run.n_fetch_paid == budget
        recovered = round((1.0 - run.distortion_recovery) * r)
        lines.append(
            f"{s:>4}  {run.n_fetch_paid:>10}  {run.n_resolve:>8}  "
            f"{recovered:>9}/{r}  {run.distortion_answer:>5.4f}  "
            f"{run.distortion_recovery:>5.4f}  {'OK' if match else 'FAIL':>5}"
        )

    d_values = [run.distortion_answer for run in runs]
    null_ok = abs(null_run.distortion_answer - 1.0) <= 1e-12
    lines.extend(
        [
            "",
            f"mean D_hat = {sum(d_values) / len(d_values):.4f}   "
            f"min = {min(d_values):.4f}   max = {max(d_values):.4f}",
            f"NullPolicy: D_hat = {null_run.distortion_answer:.4f}",
            (
                "invariants: D_hat == D_rec in all 20 runs: "
                f"{'OK' if all_match else 'FAIL'} | paid_fetch == 10 in all runs: "
                f"{'OK' if all_paid else 'FAIL'}"
            ),
            "note: sequential exclusion can make mean D_hat slightly below D_seed_inf; not a bug.",
            "log written: outputs/logs/blind_id_dryrun.txt",
        ]
    )

    text = "\n".join(lines)
    log = ROOT / "outputs" / "logs" / "blind_id_dryrun.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(text + "\n", encoding="utf-8")
    print(text)

    if not (all_match and all_paid and null_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
