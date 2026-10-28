#!/bin/bash
#SBATCH -A ACCOUNT
#SBATCH -p PARTITION
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem 16G
#SBATCH -J paper_nb_trace
#SBATCH -o analysis_out/paper_notebooks/trace/%x_%A_%a.log
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Which files does each figure notebook read? Executes it under strace (openat) into a scratch copy (the notebook
# itself is untouched) and keeps every opened file that resolves into data/ or inputs/ (or anything else outside the
# repo, conda and the system), with its real path and size: analysis_out/paper_notebooks/trace/<fig>_reads.tsv
#   sbatch --array 1-$(grep -c . paper/figures/notebooks.txt) paper/slurm/notebooks_trace.sbatch
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
REPO=$PWD
T=$REPO/analysis_out/paper_notebooks/trace
NB=${NB:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" paper/figures/notebooks.txt)}
FIG=$(basename "$NB" .ipynb)
DIR=$(dirname "$NB")
cd "$DIR"
strace -f -qq -e trace=openat -e signal=none -o "$T/${FIG}.strace" \
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3000 \
  --output-dir "$T/nb" --output "$FIG" "$(basename "$NB")"
python - "$T/${FIG}.strace" "$PWD" "$REPO" "$T/${FIG}_reads.tsv" <<'EOF'
import os, re, sys
trace, here, repo, out = sys.argv[1:]
skip = ('/proc', '/sys', '/dev', '/etc', '/usr', '/tmp', '/run', '/lib', os.path.expanduser('~/.conda'),
        os.path.expanduser('~/.cache'), os.path.expanduser('~/.local'), os.path.expanduser('~/.ipython'),
        os.path.expanduser('~/.jupyter'), os.path.expanduser('~/.config'), '/hpc/software', '/software')
seen = {}
for line in open(trace):
    m = re.search(r'openat\([^,]+, "([^"]+)", ([^)]*)\) = (\d+)', line)
    if not m or 'O_DIRECTORY' in m.group(2) or 'O_WRONLY' in m.group(2) or 'O_RDWR' in m.group(2):
        continue
    p = m.group(1)
    a = os.path.normpath(os.path.join(here, p))
    r = os.path.realpath(a)
    if r.startswith(skip) or not os.path.isfile(r) or r.endswith(('.pyc', '.py', '.so', '.ipynb')):
        continue
    rel = os.path.relpath(a, here)
    if r.startswith(os.path.realpath(repo)) and not rel.startswith(('data/', 'inputs/')) \
            and '/render/' not in r and not r.startswith(os.path.realpath(os.path.join(repo, 'analysis_out'))):
        continue                      # repo files (scripts, style, the notebook itself)
    if '/render/' in r:
        continue                      # the notebook's own outputs read back
    seen.setdefault(r, rel)
with open(out, 'w') as f:
    f.write('notebook_path\treal_path\tbytes\n')
    for r, rel in sorted(seen.items(), key=lambda x: x[1]):
        f.write(f'{rel}\t{r}\t{os.path.getsize(r)}\n')
print(f'{len(seen)} files, {sum(os.path.getsize(r) for r in seen) / 1e6:.1f} MB -> {out}')
EOF
rm -f "$T/${FIG}.strace"
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
