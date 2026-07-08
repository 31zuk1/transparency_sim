import csv
import json
from pathlib import Path

import pytest

from scripts import aggregate_arm_a, make_fig2, run_arm_a_grid
from transparency_sim.a0 import a0_exact
from transparency_sim.generator import generate_corpus
from transparency_sim.grid import (
    completed_keys,
    corpus_seed,
    load_config,
    rep_seed,
    run_key,
    sorted_cells,
)
from transparency_sim.instrument import InstrumentSpec
from transparency_sim.llm_blind_id import run_llm_blind_id
from transparency_sim.llm_client import TranscriptReplayClient, extract_openai_output_text


PILOT = Path("configs/grid_pilot.json")
REHEARSAL = Path("configs/grid_rehearsal_offline.json")


def test_load_config_validates_and_merges_duplicate_cells():
    config = load_config(PILOT)

    assert len(config.cells) == 19
    cell = next(c for c in config.cells if c.q == 150 and c.c == 0.0 and c.B == 10)
    assert cell.series == ("P1", "P3")
    cell = next(c for c in config.cells if c.q == 150 and c.c == 0.5 and c.B == 10)
    assert cell.series == ("P1", "P3")


def test_sorted_cells_and_seed_derivation_are_deterministic():
    config = load_config(REHEARSAL)
    cells = sorted_cells(config)

    assert [(c.q, c.c, c.B) for c in cells] == [(30, 0.0, 6), (30, 1.0, 6)]
    assert corpus_seed(config, 1, 0) == 425242
    assert rep_seed(1, 0, 1) == 100098


def test_run_key_format():
    cell = sorted_cells(load_config(PILOT))[0]

    assert run_key(cell, 0, 0, "live") == "q50_c0.0_B10_i0_r0_live"


def test_completed_keys_scans_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {"grid_meta": {"run_key": "a"}},
        {"no_meta": True},
        {"grid_meta": {"run_key": "b"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert completed_keys(path) == {"a", "b"}
    assert completed_keys(tmp_path / "missing.jsonl") == set()


def test_a0_arm_writes_csv_with_exact_values():
    out = Path("outputs/results/a0_grid.csv")
    if out.exists():
        out.unlink()

    assert run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "a0"]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    config = load_config(REHEARSAL)
    cell = sorted_cells(config)[0]
    seed = corpus_seed(config, 0, 0)
    corpus = generate_corpus(q=cell.q, r=config.r, c=cell.c, seed=seed)
    row = next(r for r in rows if r["q"] == "30" and r["c"] == "0.0" and r["B"] == "6")

    assert float(row["d_seed_inf"]) == pytest.approx(a0_exact(corpus, 6, "inf").distortion)


def test_a0_arm_dense_sweep_for_p3_cells():
    run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "a0"])
    rows = list(csv.DictReader(Path("outputs/results/a0_grid.csv").open(encoding="utf-8")))
    p3_bs = {
        int(r["B"]) for r in rows
        if r["q"] == "30" and r["c"] == "1.0"
    }

    assert set(range(31)) <= p3_bs


def test_offline_arm_end_to_end_and_resume():
    record = Path("outputs/runs/arm_a/arm_a_rehearsal_offline_offline.jsonl")
    if record.exists():
        record.unlink()

    assert run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "offline"]) == 0
    rows = record.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "offline"]) == 0
    assert len(record.read_text(encoding="utf-8").splitlines()) == 4


def test_offline_records_carry_grid_meta():
    record = Path("outputs/runs/arm_a/arm_a_rehearsal_offline_offline.jsonl")
    if not record.exists():
        run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "offline"])
    first = json.loads(record.read_text(encoding="utf-8").splitlines()[0])

    assert first["grid_meta"]["run_key"]
    assert first["grid_meta"]["arm"] == "offline"
    assert "cell_index" in first["grid_meta"]
    assert first["grid_meta"]["series"]


def test_aggregate_produces_runs_and_cells_csv(tmp_path):
    record = tmp_path / "records.jsonl"
    rows = [
        _record("a", 0.0, 0.5, 0.25, 5, 10, 1),
        _record("b", 0.0, 0.5, 0.25, 10, 10, 0),
    ]
    record.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert aggregate_arm_a.main(["--config", str(REHEARSAL), "--records", str(record)]) == 0
    run_rows = list(csv.DictReader(Path("outputs/results/arm_a_runs.csv").open(encoding="utf-8")))
    cell_rows = list(csv.DictReader(Path("outputs/results/arm_a_cells.csv").open(encoding="utf-8")))

    assert run_rows[0]["gap"] == "0.25"
    assert run_rows[0]["budget_utilization"] == "0.5"
    assert len(cell_rows) == 1
    assert float(cell_rows[0]["mean_gap"]) == pytest.approx(0.25)


def test_aggregate_refuses_missing_a0_coverage(tmp_path):
    record = tmp_path / "records.jsonl"
    a0 = tmp_path / "a0.csv"
    record.write_text(json.dumps(_record("a", 0.0, 0.5, 0.25, 5, 10, 1)) + "\n",
                      encoding="utf-8")
    a0.write_text(
        "config_name,q,c,corpus_seed,B,depth,d_seed_1,d_seed_inf\n"
        "x,30,0.0,999,10,inf,1.0,1.0\n",
        encoding="utf-8",
    )

    assert aggregate_arm_a.main([
        "--config", str(REHEARSAL), "--records", str(record), "--a0", str(a0),
    ]) == 2


def test_precision_table_formula():
    assert aggregate_arm_a.n_required(0.1, 0.05) == 16


def test_fig2_offline_has_watermark_and_files():
    run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "a0"])
    run_arm_a_grid.main(["--config", str(REHEARSAL), "--arm", "offline"])
    png, pdf = make_fig2.make_fig2(
        REHEARSAL,
        "outputs/runs/arm_a/arm_a_rehearsal_offline_offline.jsonl",
        "outputs/results/a0_grid.csv",
        "offline",
    )

    assert make_fig2.WATERMARK == "OFFLINE REHEARSAL - scripted client - not a result"
    assert png.exists()
    assert pdf.exists()


def test_fig2_refuses_missing_a0_coverage():
    a0_rows = [{"q": "150", "c": "0.0", "B": "10", "corpus_seed": "1"}]
    obs_rows = [{"q": 150, "c": 0.0, "B": 10, "corpus_seed": 2}]

    with pytest.raises(ValueError, match="A0 table is missing"):
        make_fig2._assert_a0_covers_observations(a0_rows, obs_rows)


def test_live_arm_refuses_without_config_flag_key_and_cap(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run_arm_a_grid.main([
        "--config", str(REHEARSAL), "--arm", "live", "--provider", "openai",
        "--model", "x",
    ]) != 0
    assert run_arm_a_grid.main([
        "--config", str(PILOT), "--arm", "live", "--provider", "openai",
        "--model", "x",
    ]) != 0
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    assert run_arm_a_grid.main([
        "--config", str(PILOT), "--arm", "live", "--provider", "openai",
        "--model", "x", "--max-new-runs", "1",
    ]) != 0


def test_extract_openai_output_text_joins_parts_with_newlines():
    class Content:
        def __init__(self, text):
            self.text = text

    class Item:
        def __init__(self, content):
            self.content = content

    class Response:
        output = [Item([Content("LIST")]), Item([Content("FETCH DOC_ABCD")])]
        output_text = "LISTFETCH DOC_ABCD"

    class Fallback:
        output = []
        output_text = "ANSWER {}"

    assert extract_openai_output_text(Response()) == "LIST\nFETCH DOC_ABCD"
    assert extract_openai_output_text(Fallback()) == "ANSWER {}"


def test_instrument_harness_version_default_and_roundtrip():
    spec = InstrumentSpec(provider="offline", model="scripted-sequential")
    old = spec.to_dict()
    old.pop("harness_version")

    assert spec.harness_version == "1.1"
    assert InstrumentSpec.from_dict(spec.to_dict()) == spec
    assert InstrumentSpec.from_dict(old).harness_version == "1.1"


def test_run_llm_blind_id_records_grid_meta_only_when_given(tmp_path):
    corpus = generate_corpus(q=30, r=5, c=0.0, seed=1)
    instrument = InstrumentSpec(provider="offline", model="replay")
    with_meta = tmp_path / "with.jsonl"
    without_meta = tmp_path / "without.jsonl"

    run_llm_blind_id(
        corpus,
        TranscriptReplayClient(["ANSWER {}"]),
        instrument,
        budget=6,
        record_path=with_meta,
        extra_meta={"run_key": "rk"},
    )
    run_llm_blind_id(
        corpus,
        TranscriptReplayClient(["ANSWER {}"]),
        instrument,
        budget=6,
        record_path=without_meta,
    )

    assert "grid_meta" in json.loads(with_meta.read_text(encoding="utf-8"))
    assert "grid_meta" not in json.loads(without_meta.read_text(encoding="utf-8"))


def _record(run_key, c, d_hat, d_rec, paid, B, protocol_errors):
    return {
        "schema_version": 1,
        "instrument": {"harness_version": "1.1"},
        "corpus": {"q": 30, "r": 5, "c": c, "seed": 123},
        "budget": B,
        "distortion_answer": d_hat,
        "distortion_recovery": d_rec,
        "n_fetch_paid": paid,
        "n_resolve": 2,
        "n_protocol_errors": protocol_errors,
        "n_sanitized_keys": 0,
        "terminated_reason": "answered",
        "n_turns": 4,
        "usage": {"input_tokens": 10, "output_tokens": 3},
        "grid_meta": {
            "run_key": run_key,
            "arm": "offline",
            "config_name": "arm_a_rehearsal_offline",
            "rep_index": 0,
            "series": ["P1"],
        },
    }


from scripts import analyze_contrasts, validate_records
from transparency_sim.grid import qc_groups


PAIRED = Path("configs/grid_rehearsal_paired.json")


def test_qc_scope_shares_corpora_across_B():
    config = load_config(PAIRED)
    cells = sorted_cells(config)

    assert config.corpus_seed_scope == "qc"
    assert qc_groups(config) == ((30, 0.0), (30, 1.0))
    assert corpus_seed(config, 1, 0) == corpus_seed(config, 2, 0)
    assert corpus_seed(config, 1, 1) == corpus_seed(config, 2, 1)
    assert corpus_seed(config, 0, 0) != corpus_seed(config, 1, 0)
    assert [(cell.q, cell.c, cell.B) for cell in cells] == [
        (30, 0.0, 6), (30, 1.0, 4), (30, 1.0, 10),
    ]


def test_cell_scope_backward_compatible():
    config = load_config(PILOT)

    assert config.corpus_seed_scope == "cell"
    for cell_index, _cell in enumerate(sorted_cells(config)):
        for instance_index in range(config.instances_per_cell):
            assert corpus_seed(config, cell_index, instance_index) == (
                config.instance_seed_base + 1000 * cell_index + instance_index
            )


def test_a0_arm_dedups_corpora_under_qc_scope():
    out = Path("outputs/results/a0_grid.csv")
    if out.exists():
        out.unlink()

    assert run_arm_a_grid.main(["--config", str(PAIRED), "--arm", "a0"]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    corpora = {(int(row["q"]), float(row["c"]), int(row["corpus_seed"])) for row in rows}
    b4_seeds = {row["corpus_seed"] for row in rows if row["c"] == "1.0" and row["B"] == "4"}
    b10_seeds = {row["corpus_seed"] for row in rows if row["c"] == "1.0" and row["B"] == "10"}

    assert len(corpora) == 4
    assert b4_seeds == b10_seeds


def test_jt_exact_permutation_separated_and_flat():
    separated = analyze_contrasts.ordered_contrast_stats([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    flat = analyze_contrasts.ordered_contrast_stats([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    assert separated["p_exact"] <= 0.01
    assert separated["exact_or_mc"] == "exact"
    assert flat["p_exact"] == 1.0


def test_p1_contrast_table_from_crafted_records(tmp_path):
    config_path = tmp_path / "p1.json"
    config_path.write_text(json.dumps({
        "config_name": "p1_crafted",
        "r": 5,
        "depth": "inf",
        "delta": 0.5,
        "instances_per_cell": 3,
        "reps_per_instance": 2,
        "instance_seed_base": 7000,
        "live_allowed": False,
        "cells": [
            {"q": 30, "c": 0.0, "B": 6, "series": ["P1"]},
            {"q": 60, "c": 0.0, "B": 6, "series": ["P1"]},
            {"q": 90, "c": 0.0, "B": 6, "series": ["P1"]},
        ],
    }), encoding="utf-8")
    config = load_config(config_path)
    run_rows = []
    for cell_index, cell in enumerate(sorted_cells(config)):
        for instance_index in range(config.instances_per_cell):
            for rep_index in range(config.reps_per_instance):
                run_rows.append({
                    "run_key": run_key(cell, instance_index, rep_index, "offline"),
                    "q": cell.q,
                    "c": cell.c,
                    "B": cell.B,
                    "corpus_seed": corpus_seed(config, cell_index, instance_index),
                    "instance_index": instance_index,
                    "rep_index": rep_index,
                    "d_hat": [0.2, 0.5, 0.8][cell_index],
                    "d_rec": [0.1, 0.4, 0.7][cell_index],
                })

    rows = analyze_contrasts.analyze_contrasts(config, run_rows, [])

    assert len(rows) == 1
    assert rows[0]["contrast_id"] == "P1_c0.0"
    assert rows[0]["n_instances_per_group"] == "3"
    assert rows[0]["delta_hat"] == pytest.approx(0.6)
    assert rows[0]["delta_hat_rec"] == pytest.approx(0.6)
    assert rows[0]["p_exact"] <= 0.01


def test_bstar_interval_and_bound_flags():
    stats = analyze_contrasts.p3_stats(
        [(4, 0.7), (10, 0.4)],
        delta=0.5,
        theory_lower=3.0,
        theory_c1_upper=8.0,
        c=1.0,
    )
    violation = analyze_contrasts.p3_stats(
        [(0, 0.8), (2, 0.4)],
        delta=0.5,
        theory_lower=3.0,
        theory_c1_upper=8.0,
        c=0.0,
    )

    assert analyze_contrasts.bstar_interval([(4, 0.7), (10, 0.4)], 0.5) == (4, 10)
    assert stats["bstar_lo"] == 4
    assert stats["bstar_hi"] == 10
    assert stats["lower_violation"] is False
    assert stats["overshoot"] == pytest.approx(2.0)
    assert violation["lower_violation"] is True


def test_contrasts_script_end_to_end_offline():
    record = Path("outputs/runs/arm_a/arm_a_rehearsal_paired_offline.jsonl")
    out = Path("outputs/results/arm_a_contrasts.csv")
    if record.exists():
        record.unlink()
    if out.exists():
        out.unlink()

    assert run_arm_a_grid.main(["--config", str(PAIRED), "--arm", "a0"]) == 0
    assert run_arm_a_grid.main(["--config", str(PAIRED), "--arm", "offline"]) == 0
    assert aggregate_arm_a.main([
        "--config", str(PAIRED), "--records", str(record), "--a0", "outputs/results/a0_grid.csv",
    ]) == 0
    assert analyze_contrasts.main([
        "--config", str(PAIRED), "--records", str(record), "--a0", "outputs/results/a0_grid.csv",
    ]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))

    assert out.exists()
    assert any(row["contrast_id"] == "P3_q30_c1.0" for row in rows)


def test_validate_records_detects_duplicates_and_schema(tmp_path):
    records = tmp_path / "records.jsonl"
    valid = _full_grid_record("q30_c0.0_B6_i0_r0_offline")
    invalid = {"schema_version": 1, "grid_meta": {"run_key": "bad"}}
    records.write_text(
        json.dumps(valid) + "\n" + json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )

    assert validate_records.main(["--records", str(records), "--config", str(PAIRED)]) == 2


def _full_grid_record(run_key_value):
    return {
        "schema_version": 1,
        "instrument": {
            "provider": "offline",
            "model": "scripted-sequential",
            "harness_version": "1.1",
        },
        "corpus": {"q": 30, "r": 5, "c": 0.0, "seed": 515151},
        "budget": 6,
        "depth": "inf",
        "conversation": [],
        "env_transcript": [],
        "answer_raw": {},
        "answer_scored": {},
        "distortion_answer": 0.5,
        "distortion_recovery": 0.5,
        "n_fetch_paid": 1,
        "n_resolve": 0,
        "n_protocol_errors": 0,
        "n_sanitized_keys": 0,
        "terminated_reason": "answered",
        "n_turns": 1,
        "usage": None,
        "timestamp_utc": "2026-07-07T00:00:00Z",
        "grid_meta": {
            "run_key": run_key_value,
            "arm": "offline",
            "config_name": "arm_a_rehearsal_paired",
            "cell_index": 0,
            "instance_index": 0,
            "rep_index": 0,
            "series": ["P1"],
        },
    }
