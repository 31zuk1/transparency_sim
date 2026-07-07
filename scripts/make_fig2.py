"""Generate Figure 2 layout for Arm A grid outputs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.grid import load_config, sorted_cells  # noqa: E402

WATERMARK = "OFFLINE REHEARSAL - scripted client - not a result"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--a0", required=True)
    parser.add_argument("--source", choices=["offline", "live"], required=True)
    args = parser.parse_args(argv)
    make_fig2(args.config, args.records, args.a0, args.source)
    return 0


def make_fig2(config_path, records_path, a0_path, source: str) -> tuple[Path, Path]:
    config = load_config(config_path)
    p3_cells = [cell for cell in sorted_cells(config) if "P3" in cell.series]
    q_values = sorted({cell.q for cell in p3_cells})
    target_q = 150 if 150 in q_values else q_values[0]
    c_values = sorted({cell.c for cell in p3_cells if cell.q == target_q})
    a0_rows = _read_a0(a0_path, target_q)
    obs = _read_records(records_path, target_q)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return _write_placeholder(source)

    fig, axes = plt.subplots(1, len(c_values), figsize=(4.2 * len(c_values), 3.8), squeeze=False)
    for ax, c in zip(axes[0], c_values):
        grouped_a0 = defaultdict(list)
        for row in a0_rows:
            if float(row["c"]) == c:
                grouped_a0[int(row["B"])].append(float(row["d_seed_inf"]))
        xs = sorted(grouped_a0)
        if xs:
            for seed in sorted({row["corpus_seed"] for row in a0_rows if float(row["c"]) == c}):
                seed_rows = sorted(
                    [row for row in a0_rows if float(row["c"]) == c and row["corpus_seed"] == seed],
                    key=lambda row: int(row["B"]),
                )
                ax.plot([int(row["B"]) for row in seed_rows],
                        [float(row["d_seed_inf"]) for row in seed_rows],
                        color="#9aa3ad", lw=0.8, alpha=0.6)
            ax.plot(xs, [statistics.fmean(grouped_a0[x]) for x in xs],
                    color="#1a5276", lw=2.0, label="calibration baseline (D_seed_inf)")

        obs_group = defaultdict(list)
        for row in obs:
            if float(row["c"]) == c:
                obs_group[int(row["B"])].append(row)
        bx = sorted(obs_group)
        if bx:
            d_hat = [statistics.fmean(float(r["d_hat"]) for r in obs_group[b]) for b in bx]
            d_rec = [statistics.fmean(float(r["d_rec"]) for r in obs_group[b]) for b in bx]
            yerr = [
                [
                    max(0.0, d_hat[i] - min(float(r["d_hat"]) for r in obs_group[b]))
                    for i, b in enumerate(bx)
                ],
                [
                    max(0.0, max(float(r["d_hat"]) for r in obs_group[b]) - d_hat[i])
                    for i, b in enumerate(bx)
                ],
            ]
            ax.errorbar(bx, d_hat, yerr=yerr, fmt="o", color="#b03a2e", label="observed D_hat")
            ax.scatter(bx, d_rec, marker="s", color="#196f3d", label="observed D_rec")
        ax.axhline(config.delta, ls=":", color="#444444", lw=1.0, label="delta")
        ax.set_title(f"q={target_q}, c={c}")
        ax.set_xlabel("B")
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("distortion (D0 = 1)")
    axes[0][-1].legend(fontsize=8, frameon=False)
    if source == "offline":
        fig.text(0.5, 0.52, WATERMARK, ha="center", va="center",
                 fontsize=18, alpha=0.25, rotation=15)
        stem = "fig2_arm_a_offline_rehearsal"
    else:
        stem = "fig2_arm_a"
    fig.tight_layout()
    out_png = ROOT / "outputs" / "figures" / f"{stem}.png"
    out_pdf = ROOT / "outputs" / "figures" / f"{stem}.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    return out_png, out_pdf


def _write_placeholder(source: str) -> tuple[Path, Path]:
    stem = "fig2_arm_a_offline_rehearsal" if source == "offline" else "fig2_arm_a"
    out_png = ROOT / "outputs" / "figures" / f"{stem}.png"
    out_pdf = ROOT / "outputs" / "figures" / f"{stem}.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    text = WATERMARK if source == "offline" else "Arm A figure placeholder"
    out_png.write_text(text + "\n", encoding="utf-8")
    out_pdf.write_text(text + "\n", encoding="utf-8")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    return out_png, out_pdf


def _read_a0(path, target_q: int) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if int(row["q"]) == target_q]


def _read_records(path, target_q: int) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        meta = record.get("grid_meta")
        if not meta or "P3" not in meta.get("series", []):
            continue
        corpus = record["corpus"]
        if int(corpus["q"]) != target_q:
            continue
        rows.append({
            "q": corpus["q"],
            "c": corpus["c"],
            "B": record["budget"],
            "d_hat": record["distortion_answer"],
            "d_rec": record["distortion_recovery"],
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
