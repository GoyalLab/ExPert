# ExPert
Snakemake pipeline to extract and combine multiple pertub-seq-like experiments. Includes downloading and preprocessing of each set.

To run the full pipeline:
1. Create mamba (conda) env with environment.yml

`mamba env create -f environment.yml`

`mamba activate ExPert`

2. (Optional) Check config/config.yaml to adjust parameters
3. Run pipeline

`snakemake --cores all`

## VAE
VAE classification model to predict source of perturbation in scRNA-seq data.
- Example use: vae/notebooks/Train.ipynb
- Main module: vae/src/modules/_jedvae.py
- Main model: vae/src/models/_jedvi.py
- Main nn elements(Encoder, DecoderSCVI, MultiHeadAttention, FunnelFCLayers, Block): vae/src/modules/_base.py
- Main training plan: vae/src/_train/plan.py::ContrastiveSupervisedTrainingPlan
- Contrastive batching: vae/src/data/_contrastive_sampler.py, vae/src/data/_contrastive_loader.py
