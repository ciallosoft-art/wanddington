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
#   experiments/15_ml_engines.sh beyond    # lgbm_rwr | mf svgp + ml_svgpacq_ucb1 (phase 2, item 4)
#   experiments/15_ml_engines.sh gdf_binary  # faithful GeneDisco with equal information: 0/1 feedback (item 3c)
#   experiments/15_ml_engines.sh gd_port   # the paper's GeneDisco port (GeneDiscoArm), re-run after the row-order fix
#   experiments/15_ml_engines.sh gd_port_tier_a  # the same on BDA's full-pool tier (Table tiers)
#   experiments/15_ml_engines.sh bnn_setrules  # BADGE / k-means-embedding on the BNN, re-run with a label-free order
#   experiments/15_ml_engines.sh gdfb_tier_a  # the equal-information GeneDisco rule chosen on the headline tier,
#                                             # applied unchanged to the full-pool tier
#   (gdf_rerun is historical: it patched the first gdf run, which the row-order fix superseded entirely)
#   experiments/15_ml_engines.sh gdf_rerun # after the float32 softuncertain fix: re-run the BNN softuncertain arms on
#                                          # every screen and the arms the crash never reached (-> *.rerun.json)
#
# TIER=full_pool runs a group on BDA's full-pool tier (core-essential genes kept, Table tiers) instead of the
# headline tier: same arms and settings, the tier_a tables, outputs in ml_engines/tier_a/ (phase 3 of the
# plan, 2026-10-13). Groups: tabular graph genept svm svm_rbf score score_hard desc bnn beyond acquisition
# gdf_binary; bnn runs the engine only (no acquisition rules on it) and beyond leaves out the GP's UCB rule.
# TIER=pairs / pairs_interaction runs the engines that take a pair's features on the combinatorial tier
# (128 pairs per round; the pair-mean table of experiment 09, or the table with the interaction features),
# outputs in ml_engines/pairs[_interaction]/ (docs/tier_extension_plan.md, part 乙4-5). Groups: tabular,
# svm (linear only), bnn (engine only), beyond (the sparse GP only), acquisition. The gene-level engines
# (RWR, GraphSAGE, GenePT, matrix factorisation, descriptors) are not defined for a pair.
set -euo pipefail
cd "$(dirname "$0")/.."

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
unset WADDINGTON_REASON_FROM_STRUCTURE PYTHONOPTIMIZE
export WADDINGTON_PARALLEL=1  # the engines already use every core; repeat-level threads would only contend
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # reproducible cuBLAS for the GNN (dense neighbour mean on the GPU)

OUT=workspace/results/ml_engines
GROUP="${1:-tabular}"
TIER="${TIER:-headline}"
DATASETS=()
case "$TIER" in
  headline) ;;
  full_pool)
    case "$GROUP" in
      tabular|graph|genept|svm|svm_rbf|score|score_hard|desc|bnn|beyond|acquisition|gdf_binary|gdf|bnn_rules|svgp_ucb) ;;
      *) echo "TIER=full_pool supports: tabular graph genept svm svm_rbf score score_hard desc bnn beyond acquisition gdf_binary gdf bnn_rules svgp_ucb" >&2; exit 2 ;;
    esac
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv"
    OUT=workspace/results/ml_engines/tier_a ;;
  pairs|pairs_interaction)
    case "$GROUP" in
      tabular|svm|bnn|beyond|acquisition|bnn_rules|svgp_ucb) ;;
      *) echo "TIER=$TIER supports: tabular svm bnn beyond acquisition bnn_rules svgp_ucb" >&2; exit 2 ;;
    esac
    PT="lgbm_training_data_horlbeck_pairs_mean"; [ "$TIER" = pairs_interaction ] && PT="lgbm_training_data_horlbeck_pairs"
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/${PT}.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/${PT}_v3.csv"
    export WADDINGTON_EXTRA_DATASETS="Horlbeck:128"
    DATASETS=(--datasets Horlbeck)
    OUT="workspace/results/ml_engines/$TIER" ;;
  *) echo "TIER must be headline, full_pool, pairs or pairs_interaction" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
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
  gdf_rerun) DET="";                         SEEDED="" ;;  # screen-parallel below
  gdf_binary) DET="";                        SEEDED="" ;;  # screen-parallel below
  gd_port|gd_port_tier_a) DET="";            SEEDED="" ;;  # explicit arm list below
  gdfb_tier_a) DET="";                       SEEDED="" ;;  # explicit arm list below
  bnn_setrules) DET="";                      SEEDED="" ;;  # explicit arm list below
  beyond)  DET="lgbm_rwr";                   SEEDED="mf svgp"; EXTRA_SEEDED="ml_svgpacq_ucb1" ;;
  bnn_rules|svgp_ucb) DET="";                SEEDED="" ;;  # other tiers: the rules on the BNN / sparse GP
  *) echo "usage: $0 [tabular|graph|genept|acquisition|svm|svm_rbf|score|score_hard|desc|bnn|gdf|gdf_rerun|beyond|gdf_binary|gd_port|gd_port_tier_a|gdfb_tier_a|bnn_setrules|bnn_rules|svgp_ucb]" >&2; exit 2 ;;
esac
if [ "$GROUP" = beyond ] && [ "$TIER" = full_pool ]; then EXTRA_SEEDED=""; fi
case "$TIER" in pairs|pairs_interaction)
  [ "$GROUP" = beyond ] && { DET=""; SEEDED="svgp"; EXTRA_SEEDED=""; } ;;
esac

arms() { for m in $1; do printf 'ml_engine_%s ml_static_%s ' "$m" "$m"; done; }

run() {  # $1 = arm list, $2 = replicates, $3 = output file
  conda run --no-capture-output -n waddington-bio python3 -m waddington_select.run_sequential \
    --arms $1 ${DATASETS[@]+"${DATASETS[@]}"} --replicates "$2" --seed-mode local --out "$3" --resume
}

if [ "$GROUP" = acquisition ]; then
  # step 6: LightGBM held fixed as a 5-member bootstrap ensemble; only the selection rule differs
  echo "== acquisition rules, 5 seeds"
  run "ml_acq_greedy ml_acq_ucb05 ml_acq_ucb1 ml_acq_thompson ml_acq_margin" 5 "$OUT/acquisition_seeded.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = bnn_rules ]; then
  # Other tiers (2026-10-16): the four acquisition rules on the Bayesian MLP; its greedy engine is group bnn.
  echo "== bnn acquisition rules ($TIER), 5 seeds"
  run "ml_bnnacq_bald ml_bnnacq_ucb1 ml_bnnacq_badge ml_bnnacq_kmeans_embedding" 5 "$OUT/bnn_rules_seeded.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = svgp_ucb ]; then
  # Other tiers: UCB on the sparse GP's latent function; its Probability-of-Hit engine is group beyond.
  echo "== sparse GP, UCB ($TIER), 5 seeds"
  run "ml_svgpacq_ucb1" 5 "$OUT/svgp_ucb_seeded.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = bnn ] && [ "$TIER" != headline ]; then
  echo "== bnn engine ($TIER), 5 seeds"
  run "ml_engine_bnn ml_static_bnn" 5 "$OUT/bnn_seeded.json"
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
if [ "$GROUP" = bnn_setrules ]; then
  # The BNN set rules took untested genes in stored row order; they now take them in a label-free order.
  # Re-run into bnn_seeded.rerun.json, which the analysis loads after bnn_seeded.json.
  echo "== bnn set rules, 5 seeds"
  run "ml_bnnacq_badge ml_bnnacq_kmeans_embedding" 5 "$OUT/bnn_seeded.rerun.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = gdfb_tier_a ]; then
  # Table tiers needs the same GeneDisco baseline on the full-pool tier. The rule is the best equal-
  # information rule on the headline tier (gdfb_achilles_bnn_coreset), not re-selected on this tier.
  mkdir -p "$OUT/port"
  echo "== equal-information GeneDisco (Achilles, BNN, coreset) on the full-pool tier, 5 seeds"
  WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv" \
  WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv" \
    run "gdfb_achilles_bnn_coreset" 5 "$OUT/port/tier_a_gdfb_coreset_seeded.json"
  echo "done -> $OUT/port"; exit 0
fi
if [ "$GROUP" = gd_port ] || [ "$GROUP" = gd_port_tier_a ]; then
  # The port used in the paper's external comparison (GeneDisco's rules on our LightGBM, binary labels),
  # re-run with the label-free pool order; kept in port/ so the engine summary does not mix it in.
  # Run it alone: its LightGBM and the forests elsewhere use n_jobs=-1 (every core, whatever
  # OMP_NUM_THREADS says), and next to other jobs the tiny fits stall in OpenMP spin-waits.
  mkdir -p "$OUT/port"
  GD_ARMS="genedisco_topuncertain genedisco_softuncertain genedisco_marginsample genedisco_coreset genedisco_kmeans_data"
  if [ "$GROUP" = gd_port ]; then
    echo "== GeneDisco port, 5 rules, 5 seeds"
    run "$GD_ARMS" 5 "$OUT/port/genedisco_port_seeded.json"
  else
    echo "== GeneDisco port on the full-pool tier, 5 rules, 5 seeds"
    WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv" \
    WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv" \
      run "$GD_ARMS" 5 "$OUT/port/tier_a_genedisco_seeded.json"
  fi
  echo "done -> $OUT/port"; exit 0
fi
if [ "$GROUP" = gdf_binary ]; then
  # item 3c: the gdf arms with a model, fed the revealed genes' 0/1 labels instead of their scores. The
  # model-free arms (random, kmeans_data) read no labels, so their gdf results stand for both settings.
  PY="$(conda run -n waddington-bio python3 -c 'import sys; print(sys.executable)')"
  ARMS="$(conda run -n waddington-bio python3 -c \
    'from waddington_select.arms.genedisco_faithful_arm import binary_arm_names; print(" ".join(binary_arm_names()))')"
  echo "== faithful GeneDisco, equal information: $ARMS"
  printf '%s\n' IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Scharenberg22 Steinhart \
    Replogle_K562_essential Replogle_K562_gwps |
    xargs -P 3 -I{} env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "$PY" -m waddington_select.run_sequential --datasets {} \
      --arms $ARMS --replicates 5 --seed-mode local --resume --out "$OUT/gdfb_seeded_{}.json"
  echo "done -> $OUT"; exit 0
fi
if [ "$GROUP" = gdf_rerun ]; then
  # The first gdf run crashed on Scharenberg22 and K562-GWPS in gdf_achilles_bnn_softuncertain (float32
  # underflow of the sampling floor; fixed in genedisco_faithful_arm). Re-run that rule for both feature sets
  # on every screen, and on the two crashed screens every arm after the last completed one
  # (gdf_achilles_bnn_topuncertain). Outputs are new *.rerun.json files; the originals are kept.
  PY="$(conda run -n waddington-bio python3 -c 'import sys; print(sys.executable)')"
  SOFT="gdf_achilles_bnn_softuncertain gdf_ours_bnn_softuncertain"
  REST="$(conda run -n waddington-bio python3 -c \
    'from waddington_select.arms.genedisco_faithful_arm import arm_names as a; n = a(); i = n.index("gdf_achilles_bnn_softuncertain"); print(" ".join(x for x in n[i + 1:] if x != "gdf_ours_bnn_softuncertain"))')"
  for ds in IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Steinhart Replogle_K562_essential; do echo "$ds $SOFT"; done > /tmp/gdf_rerun_jobs.$$
  for ds in Scharenberg22 Replogle_K562_gwps; do echo "$ds $SOFT $REST"; done >> /tmp/gdf_rerun_jobs.$$
  echo "== faithful GeneDisco re-run"; cat /tmp/gdf_rerun_jobs.$$
  xargs -P 3 -L 1 sh -c 'ds=$0; env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      '"$PY"' -m waddington_select.run_sequential --datasets "$ds" --arms "$@" --replicates 5 \
      --seed-mode local --resume --out '"$OUT"'/gdf_seeded_"$ds".rerun.json' < /tmp/gdf_rerun_jobs.$$
  rm -f /tmp/gdf_rerun_jobs.$$
  echo "done -> $OUT"; exit 0
fi
if [ -n "$DET" ]; then
  echo "== $GROUP: deterministic engines ($DET), 2 replicates"
  run "$(arms "$DET")" 2 "$OUT/${GROUP}_deterministic.json"
fi
if { [ "$GROUP" = svm ] || [ "$GROUP" = svm_rbf ]; } && [ -z "${DATASETS[*]:-}" ]; then
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
  run "$(arms "$SEEDED") ${EXTRA_SEEDED:-}" 5 "$OUT/${GROUP}_seeded.json"
fi
echo "done -> $OUT"
