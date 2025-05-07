# install pipx
pip install pipx --user
# set profile directory
profile_dir="./"

# use cookiecutter to create the profile in the config directory
template="gh:Snakemake-Profiles/slurm"
pipx run cookiecutter --output-dir "$profile_dir" "$template"