# LLM Blind-ID Live Pilot Report

Date: 2026-07-07  
Repository: `31zuk1/transparency_sim`  
Baseline commit before pilot: `7f08ff8` (`Ignore local environment files`)  
Purpose: live transport check only; not a research result.

## Scope

This pilot exercised the LLM Blind-ID text protocol against a live OpenAI
model. The goal was to confirm plumbing behavior:

- live client construction through `--live`
- protocol compliance and recovery from protocol errors
- JSONL run-record creation
- termination reasons
- token usage shape

No manuscript-facing inference should be drawn from these two runs.

## Configuration

- Provider: `openai`
- Model: `gpt-5.4-mini`
- Runs: `2`
- Corpus: `generate_corpus(q=50, r=5, c=0.5, seed=2)`
- Observer budget: `B=10`
- Depth: `inf`
- Temperature: `0.0`
- Prompt version: `blind-id-v1`
- Protocol version: `1`
- Max turns: `60`
- Max protocol errors: `5`

The OpenAI key was loaded from local `.env` via `OPENAI_API_KEY`. The key value
was not printed, logged, or committed. `.env` is ignored by Git.

## Preflight

Before the live run:

- Offline dry run passed.
- Optional LLM dependencies were installed from `requirements-llm.txt`.
- `.env` was added to `.gitignore`.
- `outputs/` was already ignored, so live JSONL records remain local unless
  explicitly added with `git add -f`.

## Commands

```bash
python scripts/run_llm_blind_id_dryrun.py
set -a; source .env; set +a
python scripts/run_llm_blind_id_dryrun.py --live --provider openai --model gpt-5.4-mini --runs 2
```

## Output Record

JSONL record:

```text
outputs/runs/llm_blind_id/live_openai_gpt-5.4-mini_20260707T085712Z.jsonl
```

The file contains the full conversation and environment transcript for each
run. It is intentionally not tracked by Git.

## Summary

| Run | Termination | D_hat | D_rec | Paid fetches | Resolves | Protocol errors | Turns | Input tokens | Output tokens |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `answered` | 0.4000 | 0.2000 | 8 | 3 | 1 | 14 | 17350 | 253 |
| 2 | `answered` | 0.4000 | 0.4000 | 8 | 2 | 1 | 13 | 15501 | 233 |

Aggregate:

- Mean `D_hat`: `0.4000`
- A0 reference on the same corpus: `D_seed_inf = 0.3106`
- Termination reasons: `{"answered": 2}`

## Protocol Observations

Both runs completed despite one protocol error each. In both cases, the model
placed more than one command-like string on the final line:

- Run 1 offending final line: `LISTLIST`
- Run 2 offending final line: `LISTFETCH DOC_7611FETCH DOC_7611`

The harness handled these as specified:

- emitted `PROTOCOL_ERROR`
- counted the violation
- continued the run
- retained the failed turn in the JSONL record

No sanitized answer keys were dropped in either run.

## Interpretation

This is a successful live plumbing check:

- The live OpenAI client path works.
- JSONL recording works.
- The protocol violation path works.
- Runs are not discarded after recoverable protocol errors.
- API key material did not enter tracked files.

The two-run sample is too small for substantive measurement. The observed
`D_hat` values and token usage should be treated as pilot diagnostics only.

## Follow-Up

Recommended next steps:

1. Decide whether prompt v1 is acceptable despite occasional command
   concatenation, or whether a v2 prompt should be drafted and frozen.
2. If keeping v1, run a slightly larger live pilot only for protocol compliance
   estimates, still not as research evidence.
3. Draft the Arm A experiment-grid specification before collecting
   manuscript-facing results.
