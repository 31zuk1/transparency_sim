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
