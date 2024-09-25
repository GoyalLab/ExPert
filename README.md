# ExPert
Snakemake pipeline to extract and combine multiple pertub-seq-like experiments. Includes downloading and preprocessing of each set.

To run the full pipeline:
1. Create mamba (conda) env with environment.yml

`mamba env create -f environment.yml`

`mamba activate ExPert`

2. (Optional) Check config/config.yaml to adjust parameters
3. Run pipeline

`snakemake --cores all`
