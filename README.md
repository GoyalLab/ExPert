# ExPert

Part A: A Snakemake pipeline for analyzing and integrating perturbation sequencing experiments.
Part B: AI framework for perturbation prediction in scRNA-seq data.

## 🚀 A: Quick Start

1. Create environment:
```bash
mamba env create -f environment.yml
mamba activate ExPert
```

2. Configure pipeline (optional):
    - Adjust parameters in `config/defaults.yaml`

3. Run pipeline:
```bash
# Locally
snakemake --cores 10 --verbose --configfile "config/defaults.yaml" --use-conda

# On SLURM
snakemake --cores 10 --verbose --configfile "config/defaults.yaml" --profile "workflow/profiles/slurm" --use-conda
```

## B: Training an ExPert Model

Our Variational Autoencoder (VAE) model predicts genetic perturbations in single-cell RNA sequencing data.

### Key Components

- **Interactive Demo**: `vae/notebooks/TrainParam.ipynb`
- **Configuration**: `resources/params/defaults.yaml`

### Core Architecture

- **Main Module**: `vae/src/modules/_jedvae.py`
- **Model Implementation**: `vae/src/models/_jedvi.py`
- **Neural Network Components**:
  - Located in `vae/src/modules/_base.py`
  - Includes: Encoder, DecoderSCVI, MultiHeadAttention, FunnelFCLayers, Block

### Training

- **Training Plan**: `vae/src/_train/plan.py::ContrastiveSupervisedTrainingPlan`
- **Contrastive Learning**:
  - Sampler: `vae/src/data/_contrastive_sampler.py`
  - Loader: `vae/src/data/_contrastive_loader.py`
