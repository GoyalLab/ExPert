# install pipx
pip install pipx --user
# set profile directory
profile_dir="/projects/b1042/GoyalLab/lschwartz/ExPert/workflow/cluster_profile"

# use cookiecutter to create the profile in the config directory
template="gh:Snakemake-Profiles/slurm"
pipx run cookiecutter --output-dir "$profile_dir" "$template"