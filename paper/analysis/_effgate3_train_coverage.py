"""Which perturbations have TRAINING cells in a fold (here EFFGATE3's fold 4, K562 screens held out)?

Writes perturbation, n_train_cells, n_train_datasets, datasets -- so drug targets can be split into
those the model learned from some non-K562 screen and those that only the held-out K562 screens carry.
"""
import argparse
import h5py, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--db', required=True); ap.add_argument('--train-idx', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
idx = np.loadtxt(a.train_idx, dtype=np.int64)
dec = lambda x: np.char.decode(x, 'utf-8') if x.dtype.kind == 'S' else x.astype(str)
with h5py.File(a.db, 'r') as f:
    ds = dec(f['obs/dataset'][:][idx]); pt = dec(f['obs/perturbation'][:][idx])
    ctrl = f['obs/is_ctrl'][:][idx].astype(bool)
d = pd.DataFrame({'perturbation': pt[~ctrl], 'dataset': ds[~ctrl]})
g = d.groupby('perturbation').agg(n_train_cells=('dataset', 'size'),
                                  n_train_datasets=('dataset', 'nunique'),
                                  datasets=('dataset', lambda s: ';'.join(sorted(set(s)))))
g.to_csv(a.out)
print(f'{len(idx):,} train cells, {len(g):,} perturbations with training cells; '
      f'datasets present: {sorted(set(ds))[:30]}')
