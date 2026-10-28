"""Cache readout queries for the ZERO-SHOT (held-out perturbation) DB.

WHY A SEPARATE SCRIPT. cache_queries.py maps labels through the registry's 8,964 TRAINED classes
and skips anything else -- which would skip every zs perturbation, since none of them are trained
(verified: 0 of 10,245 overlap). Here truth is the row in the FULL 19,895-row atom table.

ROW -> GENE is recovered from `cls_emb._atom_text`, the text bank stashed before random_init_proxies
overwrote the table: matching its rows against the source npz is exact (distance 0.0). Rows 0-8963
are the trained labels in registry order; 8964+ are the untrained remainder.

Caps cells per (dataset, perturbation) so predict() stays cheap -- a 25-cell group mean is barely
noisier than a 100-cell one and the query count is what matters, not the cell count.
"""
import argparse, glob, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--text-npz", required=True)
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import anndata as ad, h5py, torch
    from expert.models._expert import ExPert
    from expert.utils.constants import MODULE_KEYS

    mp = a.model if a.model.endswith(".pt") else max(
        glob.glob(os.path.join(a.model, "**", "model.pt"), recursive=True), key=os.path.getmtime)
    blob = torch.load(mp, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    A = sd[next(k for k in sd if k.endswith("cls_emb.embedding.weight"))].float().numpy()
    T = sd[next(k for k in sd if k.endswith("_atom_text"))].float().numpy()
    n_trained = len(np.asarray(
        blob["attr_dict"]["registry_"]["field_registries"]["labels"]["state_registry"]
        ["categorical_mapping"]))
    z = np.load(a.text_npz, allow_pickle=True)
    E = np.asarray(z["embeddings"], np.float32); genes = np.asarray(z["genes"]).astype(str)
    # exact row -> gene via the stashed text bank (hash the rows; embeddings are unique)
    key = {E[i].tobytes(): genes[i] for i in range(E.shape[0])}
    row_gene = np.array([key.get(T[i].tobytes(), "") for i in range(T.shape[0])])
    hit = int((row_gene != "").sum())
    print(f"[vocab] atom table {A.shape}, resolved {hit:,}/{T.shape[0]:,} rows to genes; "
          f"{n_trained:,} trained", flush=True)
    if hit < T.shape[0]:
        raise SystemExit("could not resolve every atom row to a gene -- refusing to guess")
    g2row = {g: i for i, g in enumerate(row_gene)}

    with h5py.File(a.adata, "r") as f:
        def col(name):
            o = f["obs"][name]
            if isinstance(o, h5py.Group):
                v = np.asarray(o["categories"].asstr()[:])[np.asarray(o["codes"][:])]
            else:
                v = np.asarray(o.asstr()[:])
            # np.char.add refuses object vs <U1 ("often lead to incorrect results"), and h5py's
            # asstr() hands back object dtype -- cast once here rather than at every use site.
            return v.astype(str)
        lab_all, ds_all = col("perturbation"), col("dataset")
    keep_m = np.array([g in g2row for g in lab_all])
    print(f"[zs] {len(lab_all):,} cells; {keep_m.sum():,} in the atom vocabulary", flush=True)

    rng = np.random.default_rng(0)
    pick = []
    kidx = np.flatnonzero(keep_m)
    k = np.char.add(np.char.add(ds_all[kidx], "|"), lab_all[kidx])
    order = np.argsort(k, kind="stable"); ks = k[order]
    start = 0
    for i in range(1, len(ks) + 1):
        if i == len(ks) or ks[i] != ks[start]:
            grp = kidx[order[start:i]]
            pick.append(grp if len(grp) <= a.cap else rng.choice(grp, a.cap, replace=False))
            start = i
    sel = np.sort(np.concatenate(pick))
    print(f"[zs] capped to {len(sel):,} cells over {len(pick):,} groups (cap {a.cap})", flush=True)

    # ---- ONE DATASET AT A TIME ------------------------------------------------------------
    # Predicting all 677,987 cells at once materialises SIX latent arrays of n_cells x n_shared:
    # at 4,096 dims that is ~22GB each in float64, ~133GB total, and it OOM-killed a 120G job.
    # The query only needs, per dataset, the dataset mean and the per-group means -- both of which
    # can be accumulated and the latents thrown away. Chunking by DATASET (not by arbitrary cell
    # blocks) is what makes that exact: `base` is a per-dataset quantity, so a dataset's cells must
    # be seen together, and they then never have to be held alongside another dataset's.
    # Cost is O(n_groups x n_shared) instead of O(n_cells x n_shared).
    mod = ExPert.load(os.path.dirname(mp), use_latest_checkpoint=True)
    mod.module.eval()
    bk = ad.read_h5ad(a.adata, backed="r")
    Q, truth, keys = [], [], []
    lab_s, ds_s = lab_all[sel], ds_all[sel]
    for di, d in enumerate(np.unique(ds_s), 1):
        rows = sel[ds_s == d]
        sub = bk[np.sort(rows)].to_memory()
        l = lab_all[np.sort(rows)]
        out = mod.predict(adata=sub.copy(), return_latents=True, obs_only=True,
                          progress_bar=False, refresh_ctrl_means=False)
        Z = np.asarray(out.get(MODULE_KEYS.Z_SHARED_KEY, out.get(MODULE_KEYS.Z_KEY)), np.float64)
        base = Z.mean(0)
        n_add = 0
        for p in np.unique(l):
            m = l == p
            if m.sum() < a.min_cells:
                continue
            Q.append((Z[m].mean(0) - base).astype(np.float32))
            truth.append(g2row[p]); keys.append(f"{d}|{p}"); n_add += 1
        print(f"  [{di:>2}/22] {d:<38} {len(rows):>7,} cells -> {n_add:>5} groups "
              f"(total {len(truth):,})", flush=True)
        del Z, out, sub
    bk.file.close(); del mod

    Q = np.stack(Q); truth = np.array(truth)
    print(f"[zs] {len(truth):,} queries; {(truth >= n_trained).sum():,} on UNTRAINED rows", flush=True)
    np.savez_compressed(a.out, Q=Q, truth=truth, keys=np.array(keys, dtype=object),
                        atoms=A.astype(np.float32), row_gene=row_gene, n_trained=n_trained)
    print(f"[zs] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
