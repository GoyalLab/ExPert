# SPDX-License-Identifier: MIT
# ExPert: Copyright (c) 2024-2026 Leon Schwartz, Northwestern University, Goyal Lab.
"""UMAP + structure of the TRAINED atoms, read straight out of model.pt.

NO adata, NO ExPert.load, NO setup. The atom table is `cls_emb.embedding.weight` in the state dict
and the label names are the registry's `labels.categorical_mapping` (already in code order), so
this is a file read and a UMAP -- seconds per arm. The first version loaded the full model and
rebuilt a control-centred pseudobulk from a BACKED h5ad in a 4,400-slice loop; it ran 15 minutes
without reaching its first print.

Only the first n_labels rows are trained. The rest of the table (19,895 at full) keeps its random
init at an indistinguishable norm -- atom/norm 0.112 vs norm_dead 0.113 -- which is how ~19.7k
random vectors once entered the OMP dictionary. Everything here slices to the live block.

Reported, because a UMAP of 200 mostly-unrelated perturbations looks like a blob whether the atoms
are good or random:
  spread     mean pairwise cosine. ~0 = spread out; ->1 = collapsed to a point.
  eff_rank   participation ratio of the singular values: how many directions are actually used.
  iso_frac   fraction of atoms whose nearest neighbour cosine is below `--iso-thr` (no neighbour).
  RANDOM     the same stats for a matched random table -- the null the picture cannot give you.
With --adata/--ctrl it additionally scores the atom geometry against the MEASURED response
(resp_rho, knn_ov), and prints each sampled atom's nearest neighbours so local structure can be
eyeballed biologically.
"""
import argparse
import os
import numpy as np


def load_atoms(model_pt):
    """(labels, atoms) for the LIVE block, straight from the checkpoint."""
    import torch
    blob = torch.load(model_pt, map_location='cpu', weights_only=False)
    sd = blob['model_state_dict']
    key = next((k for k in sd if k.endswith('cls_emb.embedding.weight')), None)
    if key is None:
        return None, None, 'no cls_emb.embedding.weight (no atom table in this checkpoint)'
    A = sd[key].float().numpy()
    reg = blob['attr_dict']['registry_']['field_registries']['labels']['state_registry']
    labels = np.asarray(reg['categorical_mapping']).astype(str)
    nl = len(labels)
    ip = blob['attr_dict'].get('init_params_', {}) or {}
    kw = {}
    for blk in ('kwargs', 'non_kwargs'):
        v = ip.get(blk)
        if isinstance(v, dict):
            kw.update(v if blk == 'non_kwargs' else
                      {k2: v2 for d in v.values() if isinstance(d, dict) for k2, v2 in d.items()})
    mode = (kw.get('extra_cls_emb_kwargs') or {}).get('adapter_mode')
    note = None
    if mode not in (None, 'identity'):
        # encode() would push these through a learned adapter, so the raw table is NOT the
        # z-space geometry the head and the OMP dictionary actually see.
        note = f'adapter_mode={mode!r}: raw table is NOT z-space; treat the plot as indicative only'
    return labels, A[:nl], note


def stats(An, k, iso_thr):
    S = An @ An.T
    n = len(An)
    iu = np.triu_indices(n, 1)
    sv = np.linalg.svd(An - An.mean(0, keepdims=True), compute_uv=False)
    eff = float((sv.sum() ** 2) / max((sv ** 2).sum(), 1e-12))
    Sd = S - np.eye(n) * 9
    nn = Sd.max(1)
    return dict(spread=float(S[iu].mean()), eff_rank=eff,
                iso_frac=float((nn < iso_thr).mean()), nn_mean=float(nn.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="model.pt paths OR dirs containing one")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--out", default="analysis_out/atom_umap")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--iso-thr", type=float, default=0.2)
    ap.add_argument("--show-nbrs", type=int, default=8, help="atoms whose neighbours are printed")
    ap.add_argument("--adata", default=None, help="optional: score atoms vs MEASURED response")
    ap.add_argument("--ctrl", default=None)
    ap.add_argument("--label-key", default="perturbation")
    ap.add_argument("--batch-key", default="dataset")
    a = ap.parse_args()

    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    os.makedirs(a.out, exist_ok=True)

    R = perts = None
    if a.adata and a.ctrl:
        import anndata as ad
        from expert.analysis._pbulk_fast import response_matrix
        print('[resp] building control-centred response (one-pass sparse)...', flush=True)
        perts, R = response_matrix(ad.read_h5ad(a.adata), ad.read_h5ad(a.ctrl),
                                   a.label_key, a.batch_key)
        Rn = R - R.mean(0, keepdims=True)
        Rn = Rn / np.clip(np.linalg.norm(Rn, axis=1, keepdims=True), 1e-12, None)
        Sr = Rn @ Rn.T
        rid = {p: i for i, p in enumerate(perts)}
        print(f'[resp] {len(perts)} perturbations', flush=True)

    rows = []
    for i, m in enumerate(a.models):
        mp = m if m.endswith('.pt') else None
        if mp is None:
            c = glob.glob(os.path.join(m, '**', 'model.pt'), recursive=True)
            if not c:
                print(f'{m}: no model.pt'); continue
            mp = max(c, key=os.path.getmtime)
        name = (a.names[i] if a.names and i < len(a.names)
                else os.path.basename(m.rstrip('/')) or f'arm{i}')
        labels, A, note = load_atoms(mp)
        if A is None:
            print(f'{name}: {note}'); continue
        if note:
            print(f'{name}: WARNING {note}')
        An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        st = stats(An, a.k, a.iso_thr)
        rng = np.random.default_rng(0)
        Rr = rng.normal(size=A.shape)
        Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
        stn = stats(Rr, a.k, a.iso_thr)

        extra = ''
        if R is not None:
            from scipy.stats import spearmanr
            common = [j for j, p in enumerate(labels) if p in rid]
            ja = np.array(common); jr = np.array([rid[labels[j]] for j in common])
            Sa_c, Sr_c = (An[ja] @ An[ja].T), Sr[np.ix_(jr, jr)]
            iu = np.triu_indices(len(ja), 1)
            rho = float(spearmanr(Sa_c[iu], Sr_c[iu]).statistic)
            kk = min(a.k, len(ja) - 1)
            na = np.argsort(-(Sa_c - np.eye(len(ja)) * 9), axis=1)[:, :kk]
            nr = np.argsort(-(Sr_c - np.eye(len(ja)) * 9), axis=1)[:, :kk]
            ov = float(np.mean([len(set(na[x]) & set(nr[x])) / kk for x in range(len(ja))]))
            st['resp_rho'] = rho; st['knn_ov'] = ov; st['knn_chance'] = kk / len(ja)
            extra = f"  resp_rho {rho:+.4f}  knn_ov {ov:.4f} (chance {kk/len(ja):.4f})"

        rows.append((name, st, stn))
        print(f"{name:<18} spread {st['spread']:+.4f} (rand {stn['spread']:+.4f})  "
              f"eff_rank {st['eff_rank']:.1f} (rand {stn['eff_rank']:.1f})  "
              f"iso {st['iso_frac']:.3f} (rand {stn['iso_frac']:.3f}){extra}", flush=True)

        if a.show_nbrs:
            Sd = An @ An.T - np.eye(len(An)) * 9
            sel = np.random.default_rng(0).choice(len(An), min(a.show_nbrs, len(An)), replace=False)
            print(f'   nearest neighbours (do these make biological sense?):')
            for x in sel:
                nb = np.argsort(-Sd[x])[:5]
                print(f"     {labels[x]:<12} -> " +
                      ", ".join(f"{labels[y]}({Sd[x, y]:+.2f})" for y in nb))

        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
                        random_state=0).fit_transform(An)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(emb[:, 0], emb[:, 1], s=16, alpha=0.85, linewidths=0)
        t = (f"{name}  trained atoms (n={len(An)})\nspread {st['spread']:+.3f} "
             f"(rand {stn['spread']:+.3f})  eff_rank {st['eff_rank']:.1f}")
        if 'resp_rho' in st:
            t += f"\nresp_rho {st['resp_rho']:+.3f}  knn_ov {st['knn_ov']:.3f}"
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        for ext in ('svg', 'png'):
            fig.savefig(os.path.join(a.out, f'atoms_{name}.{ext}'), dpi=300)
        plt.close(fig)

    if not rows:
        raise SystemExit('no atom table found in any input -- nothing plotted (exiting nonzero)')
    print(f'\nplots -> {a.out} (svg + png @300dpi)')
    print('READ: spread near the random null and eff_rank high = the atoms are spread out, not')
    print('  collapsed. resp_rho/knn_ov >> 0 = the geometry recovers MEASURED response structure;')
    print('  on 200 mostly-unrelated perturbations expect these near 0 and judge full scale.')


if __name__ == "__main__":
    main()
