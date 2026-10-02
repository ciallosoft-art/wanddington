#!/usr/bin/env bash
# Experiment 15 — alternative ML engines, ML only (no LLM calls). docs/ml_engines_plan.md, steps 2-3.
#
# Every arm is the C-arm's own ML engine (MLEngineArm: same metadata DepMap feature policy, same anchor
# drop, same leave-one-screen-out rows, same x10 online weighting) with only the model swapped:
#   ml_engine_<model>  retrained on each round's revealed genes
#   ml_static_<model>  the round-0 cross-experiment prior, never retrained
# Deterministic engines run two replicates, which must come out identical (checked in step 5); seeded
# engines run five seeds via --seed-mode local (replicate i uses seed i).
#
#   experiments/15_ml_engines.sh tabular   # lightgbm logreg lambdarank | rf mlp
#   experiments/15_ml_engines.sh graph     # rwr | gnn
#   experiments/15_ml_engines.sh genept    # genept_lr genept_lgbm (needs the GenePT download, plan step 4)
#   experiments/15_ml_engines.sh acquisition  # ml_acq_{greedy,ucb05,ucb1,thompson,margin} (plan step 6)
#   experiments/15_ml_engines.sh svm       # svm_linear | svm_rbf (Nystroem), added after the first report
#   experiments/15_ml_engines.sh score     # lgbm_score lgbm_score_xent: learn from training screens' continuous
#                                          # scores instead of their hit labels (phase 2, item 1)
#   experiments/15_ml_engines.sh score_hard  # the same with revealed genes on hard 0/1 targets (post hoc)
#   experiments/15_ml_engines.sh desc      # lgbm_depmap_pca lgbm_string_emb lgbm_desc: + 32-dim DepMap profile /
#                                          # STRING eigenmap descriptors (phase 2, item 2)
#   experiments/15_ml_engines.sh bnn       # GeneDisco's Bayesian MLP in our protocol + bald/ucb1/badge/kmeans_embedding
#   experiments/15_ml_engines.sh gdf       # faithful GeneDisco baseline (gdf_* arms), screen-parallel (phase 2, item 3)
set -euo pipefail
cd "$(dirname "$0")/.."

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
unset WADDINGTON_REASON_FROM_STRUCTURE PYTHONOPTIMIZE
export WADDINGTON_PARALLEL=1  # the engines already use every core; repeat-level threads would only contend
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # reproducible cuBLAS for the GNN (dense neighbour mean on the GPU)

OUT=workspace/results/ml_engines
mkdir -p "$OUT"

GROUP="${1:-tabular}"
case "$GROUP" in
  tabular) DET="lightgbm logreg lambdarank"; SEEDED="rf mlp" ;;
  graph)   DET="rwr";                        SEEDED="gnn" ;;
  genept)  DET="genept_lr genept_lgbm";      SEEDED="" ;;
  acquisition) DET=""; SEEDED="" ;;  # explicit arm list below
  svm)     DET="svm_linear";                 SEEDED="" ;;  # svm_rbf runs screen-parallel below
  svm_rbf) DET="";                           SEEDED="" ;;  # only the screen-parallel svm_rbf part
  score)   DET="lgbm_score lgbm_score_xent"; SEEDED="" ;;
  score_hard) DET="lgbm_score_hard lgbm_score_xent_hard"; SEEDED="" ;;
  desc)    DET="lgbm_depmap_pca lgbm_string_emb lgbm_desc"; SEEDED="" ;;
  bnn)     DET="";                           SEEDED="" ;;  # explicit arm list below
  gdf)     DET="";                           SEEDED="" ;;  # screen-parallel below
  *) echo "usage: $0 [tabular|graph|genept|acquisition|svm|svm_rbf|score|score_hard|desc|bnn|gdf]" >&2; exit 2 ;;
esac

arms() { for m in $1; do printf 'ml_engine_%s ml_static_%s ' "$m" "$m"; done; }

run() {  # $1 = arm list, $2 = replicates, $3 = output file
  conda run --no-capture-output -n waddington-bio python3 -m waddington_select.run_sequential \
    --arms $1 --replicates "$2" --seed-mode local --out "$3" --resume
}

if [ "$GROUP" = acquisition ]; then
  # step 6: LightGBM held fixed as a 5-member bootstrap ensemble; only the selection rule differs
  echo "== acquisition rules, 5 seeds"
  run "ml_acq_greedy ml_acq_ucb05 ml_acq_ucb1 ml_acq_thompson ml_acq_margin" 5 "$OUT/acquisition_seeded.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = bnn ]; then
  # phase 2 item 3b: the Bayesian MLP engine (greedy on its MC mean, online and static) and four rules on it
  echo "== bnn engine and acquisition rules, 5 seeds"
  run "ml_engine_bnn ml_static_bnn ml_bnnacq_bald ml_bnnacq_ucb1 ml_bnnacq_badge ml_bnnacq_kmeans_embedding" 5 \
    "$OUT/bnn_seeded.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = gdf ]; then
  # phase 2 item 3a: every faithful-GeneDisco arm, 5 seeds. k-means on the 1,178-dim Achilles profile takes
  # ~25 s per round on a full screen, so screens run three at a time with their threads capped.
  PY="$(conda run -n waddington-bio python3 -c 'import sys; print(sys.executable)')"
  ARMS="$(conda run -n waddington-bio python3 -c \
    'from waddington_select.arms.genedisco_faithful_arm import arm_names; print(" ".join(arm_names()))')"
  echo "== faithful GeneDisco: $ARMS"
  printf '%s\n' IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Scharenberg22 Steinhart \
    Replogle_K562_essential Replogle_K562_gwps |
    xargs -P 3 -I{} env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "$PY" -m waddington_select.run_sequential --datasets {} \
      --arms $ARMS --replicates 5 --seed-mode local --resume --out "$OUT/gdf_seeded_{}.json"
  echo "done -> $OUT"; exit 0
fi
if [ -n "$DET" ]; then
  echo "== $GROUP: deterministic engines ($DET), 2 replicates"
  run "$(arms "$DET")" 2 "$OUT/${GROUP}_deterministic.json"
fi
if [ "$GROUP" = svm ] || [ "$GROUP" = svm_rbf ]; then
  # The Nystroem-RBF SVM spends ~30 s per fit in single-threaded liblinear (200 landmarks). Screens are
  # independent, so run one single-threaded process per screen, three at a time: five at a time with
  # 500 landmarks contended for memory bandwidth and ran ~4x slower (see ml_engines/_aborted_svm_rbf_500).
  PY="$(conda run -n waddington-bio python3 -c 'import sys; print(sys.executable)')"
  echo "== svm: seeded engine (svm_rbf), 5 seeds, screen-parallel"
  printf '%s\n' IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Scharenberg22 Steinhart \
    Replogle_K562_essential Replogle_K562_gwps |
    xargs -P 3 -I{} env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PY" -m waddington_select.run_sequential --datasets {} \
      --arms ml_engine_svm_rbf ml_static_svm_rbf --replicates 5 --seed-mode local --resume \
      --out "$OUT/svm_seeded_{}.json"
  echo "done -> $OUT"; exit 0
fi
if [ -n "$SEEDED" ]; then
  echo "== $GROUP: seeded engines ($SEEDED), 5 seeds"
  run "$(arms "$SEEDED")" 5 "$OUT/${GROUP}_seeded.json"
fi
echo "done -> $OUT"
