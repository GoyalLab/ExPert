"""Cap an ID split to <=N cells per (dataset, perturbation), mirroring how the OOD cache was built.

The ID/OOD comparison only means something if the queries are constructed identically; the OOD
cache used ood_test_cap25, and an uncapped id_val would both cost a 980k-cell predict() and give
the well-sampled classes far more influence than they have on the OOD side.
"""
import gzip, sys
import numpy as np, h5py

adata, split_in, split_out, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# optional: keep only groups with >= min_cells cells, then a random n_groups of them (depth benchmarks)
min_cells = int(sys.argv[5]) if len(sys.argv) > 5 else 0
n_groups = int(sys.argv[6]) if len(sys.argv) > 6 else 0
with gzip.open(split_in, 'rt') as fh:
    idx = np.array([int(l) for l in fh if l.strip()], dtype=np.int64)
with h5py.File(adata, 'r') as f:
    o = f['obs']
    def col(n):
        g = o[n]
        return (np.asarray(g['categories'].asstr()[:])[np.asarray(g['codes'][:])].astype(str)
                if isinstance(g, h5py.Group) else np.asarray(g.asstr()[:]).astype(str))
    lab = col('perturbation')[idx]; ds = col('dataset')[idx]
key = np.char.add(np.char.add(ds, '|'), lab)
rng = np.random.default_rng(0)
keep = []
order = np.argsort(key, kind='stable')
ks, st, groups = key[order], 0, []
for i in range(1, len(ks) + 1):
    if i == len(ks) or ks[i] != ks[st]:
        if i - st >= min_cells:
            groups.append(order[st:i])
        st = i
if n_groups and len(groups) > n_groups:
    groups = [groups[j] for j in np.sort(rng.choice(len(groups), n_groups, replace=False))]
for g in groups:
    keep.append(g if len(g) <= cap else rng.choice(g, cap, replace=False))
keep = np.sort(idx[np.concatenate(keep)])
with gzip.open(split_out, 'wt') as fh:
    fh.write('\n'.join(map(str, keep.tolist())) + '\n')
print(f'{split_in} -> {split_out}: {len(idx):,} cells -> {len(keep):,} '
      f'({len(groups):,} of {len(np.unique(key)):,} groups kept, min {min_cells}, cap {cap})')
