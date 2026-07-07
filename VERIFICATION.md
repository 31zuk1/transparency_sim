# VERIFICATION v2 — 第2次修正の復元マージ後の状態

Codex 実装(環境 API・scoring・Blind-ID・dryrun・テスト26件)と第2次修正(ef_one_hop・全グラフ列挙テスト・図v2・diff提案v2)を統合した状態の検証記録。

## ファイル一覧
```
./.gitignore
./README.md
./conftest.py
./manuscript_diff_recommendations.md
./requirements.txt
./scripts/regenerate_fig1.py
./scripts/run_a0_smoke.py
./scripts/run_blind_id_dryrun.py
./src/transparency_sim/__init__.py
./src/transparency_sim/a0.py
./src/transparency_sim/blind_id.py
./src/transparency_sim/corpus.py
./src/transparency_sim/environment.py
./src/transparency_sim/generator.py
./src/transparency_sim/plots.py
./src/transparency_sim/scoring.py
./src/transparency_sim/theory.py
./tests/test_a0.py
./tests/test_blind_id.py
./tests/test_environment.py
./tests/test_generator.py
./tests/test_scoring.py
./tests/test_theory.py
```

## pytest(51件)
```
============================== 51 passed in 0.26s ==============================
```
完全ログ: outputs/logs/pytest.log

## A0 理論整合(端点・閉形式照合)
```
c=0, d=1: A0 D = 0.800000000000  theory = 0.800000000000  |diff| = 0.00e+00
c=0, d=inf: A0 D = 0.800000000000  theory = 0.800000000000  |diff| = 0.00e+00
c=1, d=1: A0 D = 0.310562782005  Pr(m=0) = 0.310562782005  |diff| = 2.78e-16
c=1, d=inf: A0 D = 0.310562782005  Pr(m=0) = 0.310562782005  |diff| = 2.78e-16
graph-avg vs closed form c=0.3: 1.416727272727 vs 1.416727272727  |diff| = 6.66e-16
graph-avg vs closed form c=0.7: 1.911272727273 vs 1.911272727273  |diff| = 6.66e-16
```

## Blind-ID dryrun
```
Blind-ID dry run (scripted sequential policy; no LLM in this round)
corpus: q=50 r=5 c=0.5 seed=2 | observer: B=10 depth=inf kappa=0
A0 reference on the same corpus: D_seed_1 = 0.4397  D_seed_inf = 0.3106

seed  paid_fetch  resolves  recovered/r  D_hat  D_rec  match
   0          10         4          5/5  0.0000  0.0000     OK
   1          10         4          5/5  0.0000  0.0000     OK
   2          10         4          5/5  0.0000  0.0000     OK
   3          10         0          0/5  1.0000  1.0000     OK
   4          10         4          5/5  0.0000  0.0000     OK
   5          10         4          5/5  0.0000  0.0000     OK
   6          10         4          5/5  0.0000  0.0000     OK
   7          10         4          5/5  0.0000  0.0000     OK
   8          10         4          5/5  0.0000  0.0000     OK
   9          10         4          5/5  0.0000  0.0000     OK
  10          10         0          0/5  1.0000  1.0000     OK
  11          10         4          5/5  0.0000  0.0000     OK
  12          10         4          5/5  0.0000  0.0000     OK
  13          10         0          0/5  1.0000  1.0000     OK
  14          10         4          5/5  0.0000  0.0000     OK
  15          10         0          0/5  1.0000  1.0000     OK
  16          10         0          0/5  1.0000  1.0000     OK
  17          10         4          5/5  0.0000  0.0000     OK
  18          10         0          0/5  1.0000  1.0000     OK
  19          10         4          5/5  0.0000  0.0000     OK

mean D_hat = 0.3000   min = 0.0000   max = 1.0000
NullPolicy: D_hat = 1.0000
invariants: D_hat == D_rec in all 20 runs: OK | paid_fetch == 10 in all runs: OK
note: sequential exclusion can make mean D_hat slightly below D_seed_inf; not a bug.
log written: outputs/logs/blind_id_dryrun.txt
```

## 実行コマンド(すべて数秒)
pip install -r requirements.txt / pytest / python scripts/regenerate_fig1.py / python scripts/run_a0_smoke.py / python scripts/run_blind_id_dryrun.py
