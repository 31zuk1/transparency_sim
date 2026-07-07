# transparency_sim

Companion code for draft v0.4, *More Disclosure, Less Transparency: Measuring
Substantive Transparency for Resource-Bounded Observers*.

Scope of this round (deliberately minimal):

1. **Figure 1** — the two-sided budget bounds of Proposition 3 (§4.5, §5.6).
   Panel (b) draws exactly three lines against the dilution ratio ρ = q/r:
   the universal lower bound `(q−r+1)(1−α^(1/r))`, the achievable upper bound
   at complete connectivity `q(1−α^(1/r))`, and the no-structure requirement
   under the linear loss of Assumption 4, `B*(δ; q, 0) = q(1−α)`.
2. **Synthetic environment generator** (`generator.py`) — small, inspectable
   corpora satisfying Assumptions 1–4 of the draft, with per-instance leak
   checks (no Y0 value in distractors; references among core documents only;
   ids and display order uninformative; exactly one Y0 component per core
   document, enforcing the linear recovery–distortion map by construction).
   Fully deterministic given `(q, r, c, seed)`. No LLM is involved.
3. **A0 baseline** (`a0.py`) — the scripted seed baseline. It computes,
   exactly, the achieved distortion `D_seed_d` of the batch-then-track
   restricted policy class Π^seed on one fixed corpus (subset enumeration
   over the core set; depths `d = 1` and `d = "inf"`). A0 is a calibration
   baseline for the theory surface. The depth-1 closed form of eq. (3) is a
   graph-generation average; A0 on one corpus is the conditional value on
   the realized graph, so cross-checks against eq. (3) average over graphs
   (an exact full-graph enumeration test at r = 3 is included).
   A0 is **not** the definitional infimum
   D* over the full adaptive class Π_θ, and it never reads document bodies.

Out of scope in this round: Figure 2, any LLM arm (Blind-ID / Metadata /
Semantic-search), human validity, bilingual corpora, large simulation grids.

## Commands

```
pip install -r requirements.txt
pytest
python scripts/regenerate_fig1.py
python scripts/run_a0_smoke.py
```

All commands finish in seconds. Figures land in `outputs/figures/`,
the smoke log in `outputs/logs/`.

## Layout

```
src/transparency_sim/   theory.py  corpus.py  generator.py  a0.py  plots.py
scripts/                regenerate_fig1.py  run_a0_smoke.py
tests/                  test_theory.py  test_generator.py  test_a0.py
outputs/                figures/  logs/
manuscript_diff_recommendations.md
```

## Environment API and Blind-ID arm (scripted dry run; no LLM yet)

`environment.py` exposes the sequential-fetch interface that enforces the
resource profile at the boundary: `list_ids` (anonymous ids only), `fetch`
(costs 1 unit of the direct-acquisition budget B), and `resolve` (free at
kappa = 0, depth-capped at d in {1, "inf"}). The Blind-ID observer class sees
nothing but anonymous ids before fetching; metadata and search are *not*
part of this class and will be separate observer classes later.
`scoring.py` scores structured answers against the generator's ground-truth
key deterministically (normalized exact match; the linear map of
Assumption 4). No model ever acts as a judge. `blind_id.py` provides
scripted policies and a harness; `scripts/run_blind_id_dryrun.py` runs a
seeded dry run and cross-checks answer-based distortion against
recovery-based distortion on every run. No LLM is called in this round.

## LLM Blind-ID arm (opt-in live runs; offline by default)

`llm_blind_id.py` drives the Blind-ID environment through a strict,
provider-neutral text protocol (LIST / FETCH / RESOLVE / ANSWER, one command
per reply). The instrument specification I (provider, model, temperature,
prompt and protocol versions, turn and violation caps) is frozen in
`instrument.py` and embedded in every JSONL run record together with the
full conversation, the environment transcript, the deterministic score, and
the termination reason; failed runs are recorded, never discarded. Tests and
the default dry run use offline clients only -- `SequentialScriptClient`
replays the scripted sequential policy through the protocol and must match
the direct policy run exactly (transport fidelity). Live calls require an
explicit `--live` flag plus a provider key in the environment, are capped,
and never appear in tests or CI. API keys are read from environment
variables and never logged.

## Arm A experiment grid (offline rehearsal now; live collection by operator)

`grid.py` and `scripts/run_arm_a_grid.py` define the preregistration-ready
grid: series P1 (monotonicity in q), P2 (equal-growth-rate erosion), and P3
(the two-sided budget bracket at q = 150). The `a0` arm computes the exact
calibration surface (D_seed_1, D_seed_inf) for every corpus instance,
including a dense B sweep for P3; the `offline` arm rehearses the entire
grid through the text protocol with the scripted client; the `live` arm is
resumable, capped by --max-new-runs, triple-guarded, and never runs in
tests or CI. `aggregate_arm_a.py` emits per-run and per-cell tables with
both distortions, their gap, budget utilization, and the precision table
that fixes instance counts for preregistration. `make_fig2.py` renders the
figure-2 layout; offline sources are watermarked as rehearsal and are not
results. Finalized records are promoted to the tracked `data/arm_a/`
directory (see RUNBOOK_ARM_A.md).
