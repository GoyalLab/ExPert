# Paths for the deployment-panel prior screen (screen.py / plot.py). Sourced by paper/slurm/fig3*.sbatch from the
# repository root; the data roots come from paper/slurm/paths.sh (set them there or in paths.local.sh).
source paper/slurm/paths.sh
PY=${PY:-python3}
S=$REPO/paper/analysis/readout/prior_screen/screen.py
P=$REPO/paper/analysis/readout/prior_screen/plot.py
DB=$DL_ROOT/data/store_out/1e6bafc04836e06a                         # ExPert training database
MODEL=$EXPERT_ROOT/models/v0.1.7/EFFGATE3/run_0/config/ExPert_v0/model/model.pt   # HF: GoyalLab/ExPert-paper-models EFFGATE3/
ATOMS=${ATOMS:-H4}                                                   # atoms + queries being scored
ZS_PANEL=${ZS_PANEL:-cand}                                           # ranking panel: cand = untrained candidates only,
                                                                     # full = whole deployed vocabulary (predict(obs_only=False))
TAG=${TAG:-$ATOMS}                                                   # output subdir under $OUT (use ${ATOMS}_full for the full panel)
OUT=$REPO/analysis_out/prior_screen
WORK=${PRIOR_SCREEN_WORK:-$REPO/analysis_out/prior_screen_work}      # large caches
EMB=${GENE_EMB_DIR:-$REPO/resources/embeddings}                      # Zenodo "ExPert - gene embeddings", unpacked
PANEL=$EMB/opus-4.8/qwen3-embed-8b/gene_embeddings.npz               # the EFFGATE3 text bank = gene panel
ZS_CACHE=$REPO/analysis_out/qcache/${ATOMS}_zs.npz
PATHS="--emb-dir $EMB --prior-dir $EMB/priors --genept-dir $REPO/resources/gene_embeddings/GenePT_emebdding_v2 \
 --random-prior $EMB/random_init/gene_embeddings_random.npz --panel $PANEL --out-dir $OUT --work-dir $WORK \
 --gene-sets-dir $OUT/gene_sets"
MODEL_DIR=$(dirname $MODEL)
RUN_CONFIG=$EXPERT_ROOT/models/v0.1.7/EFFGATE3/run_0/config.yaml
CAPTURE_TAG=EFFGATE3                                                 # model whose refit defines the ExPert fill recipe
CAPTURE=$WORK/$CAPTURE_TAG/capture.npz
CAPTURE_ROWS=$REPO/analysis_out/qcache/${CAPTURE_TAG}_zs.npz          # zs cache of that model (row order check)
SPLIT_META=$DB/splits/full/metadata.json
FOLD=4
ZS_EXTRA=${ZS_EXTRA:-}                                               # extra zs flags (e.g. --reuse-fills for a second panel)
ZS_ADATA=$DB/database_perturbed_zs.h5ad
ZS_ARGS="--zs-cache $ZS_CACHE --zs-adata $ZS_ADATA --capture $CAPTURE --capture-rows $CAPTURE_ROWS \
 --split-meta $SPLIT_META --fold $FOLD --tag $TAG --zs-panel $ZS_PANEL --pred-tag $ATOMS"
