"""Compute the readout's group pseudobulk QUERIES once and cache them.

WHY. Every atom-substitution experiment re-runs predict() on 251,525 cells (~25 min) only to
rebuild the SAME queries. The queries depend on the model, not on the atom table, so an experiment
that only swaps atoms can reuse them: cache Q once and every later variant is pure numpy in
seconds. This is what makes "try a different map / prior / objective" cheap instead of hourly.

Queries are built exactly as ood_readout_probe does: per-(dataset, perturbation) pseudobulk minus
the eval-split dataset mean, min_cells 5.
"""
import argparse, glob, gzip, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--split", default="ood_test_cap25.txt.gz")
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-cells", type=int, default=0,
                    help="also save this many random single cells (z minus their dataset base) to --cells-out")
    ap.add_argument("--cells-out", default=None)
    ap.add_argument("--pool-sizes", type=int, nargs="*", default=None,
                    help="instead of ONE pool of all cells per dataset x perturbation, one random pool per size "
                         "(0 = all cells) for every group with at least that many cells (fixed seed)")
    # OPT-IN, defaults keep the original file format byte-compatible.
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16", "auto"),
                    help="storage dtype for Q. 'auto' = float16 only if the float16 queries reproduce "
                         "--ranks-csv's top1/top10 membership exactly, else float32")
    ap.add_argument("--no-atoms", action="store_true",
                    help="do not store the atom table (it is a copy of the checkpoint; load_atoms reads it)")
    ap.add_argument("--ranks-csv", default=None,
                    help="dumped <ARM>_ranks.csv from ood_readout_probe: recompute ranks from these queries "
                         "and report exact agreement (float64 / float32 / float16 queries)")
    a = ap.parse_args()

    import anndata as ad
    from expert.models._expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    from expert.analysis.atom_umap import load_atoms

    with gzip.open(os.path.join(a.fold_dir, a.split), "rt") as fh:
        idx = np.array([int(l) for l in fh if l.strip()], dtype=np.int64)
    bk = ad.read_h5ad(a.adata, backed="r")
    sub = bk[np.sort(idx)].to_memory(); bk.file.close()
    lab = sub.obs["perturbation"].astype(str).to_numpy()
    ds = sub.obs["dataset"].astype(str).to_numpy()

    mp = a.model if a.model.endswith(".pt") else max(
        glob.glob(os.path.join(a.model, "**", "model.pt"), recursive=True), key=os.path.getmtime)
    labels, A, _ = load_atoms(mp)
    l2i = {p: i for i, p in enumerate(labels)}

    mod = ExPert.load(os.path.dirname(mp), use_latest_checkpoint=True)
    mod.module.eval()
    out = mod.predict(adata=sub.copy(), return_latents=True, obs_only=True,
                      progress_bar=False, refresh_ctrl_means=False)
    Z = np.asarray(out.get(MODULE_KEYS.Z_SHARED_KEY, out.get(MODULE_KEYS.Z_KEY)), np.float64)
    del mod

    if a.save_cells:
        rc = np.random.default_rng(1)
        ok = np.array([p in l2i for p in lab])
        bases = {d: Z[ds == d].mean(0) for d in np.unique(ds)}
        pick = np.sort(rc.choice(np.flatnonzero(ok), min(a.save_cells, int(ok.sum())), replace=False))
        Zc = np.stack([Z[i] - bases[ds[i]] for i in pick]).astype(np.float32)
        np.savez_compressed(a.cells_out, Z=Zc, truth=np.array([l2i[lab[i]] for i in pick]),
                            dataset=ds[pick].astype(str), key=np.char.add(np.char.add(ds[pick].astype(str), "|"),
                                                                          lab[pick].astype(str)))
        print(f"[cells] {len(pick):,} single cells -> {a.cells_out}", flush=True)
    Q, truth, keys, ncell = [], [], [], []
    rng_pool, pool_req = np.random.default_rng(0), []
    for d in np.unique(ds):
        md = ds == d
        base = Z[md].mean(0)
        for p in np.unique(lab[md]):
            sel = md & (lab == p)
            if sel.sum() < a.min_cells or p not in l2i:
                continue
            if a.pool_sizes:
                idx_ = np.flatnonzero(sel)
                for n_ in a.pool_sizes:
                    if n_ and len(idx_) < n_:
                        continue
                    pick = idx_ if not n_ else rng_pool.choice(idx_, n_, replace=False)
                    Q.append(Z[pick].mean(0) - base); truth.append(l2i[p])
                    keys.append(f"{d}|{p}"); ncell.append(len(pick)); pool_req.append(n_)
                continue
            Q.append(Z[sel].mean(0) - base); truth.append(l2i[p])
            keys.append(f"{d}|{p}"); ncell.append(int(sel.sum()))
    Q = np.stack(Q)                                   # float64, exactly what the readout scores
    truth_a = np.array(truth)
    dt = np.float32 if a.dtype in ("float32", "auto") else np.float16
    if a.ranks_csv:
        import csv
        with open(a.ranks_csv) as fh:
            ref = {(r["dataset"], r["perturbation"]): int(r["rank"]) for r in csv.DictReader(fh)}
        kk = [tuple(k.split("|", 1)) for k in keys]
        miss = sum(k not in ref for k in kk)
        rref = np.array([ref.get(k, -1) for k in kk])
        An = np.asarray(A, np.float32)
        An = An / np.clip(np.linalg.norm(An, axis=1, keepdims=True), 1e-12, None)

        def _ranks(Qx):
            Qn = Qx / np.clip(np.linalg.norm(Qx, axis=1, keepdims=True), 1e-12, None)
            out = np.empty(len(truth_a), np.int64)
            for s in range(0, len(truth_a), 2000):
                S = Qn[s:s + 2000] @ An.T
                t = S[np.arange(S.shape[0]), truth_a[s:s + 2000]]
                out[s:s + 2000] = (S > t[:, None]).sum(1)
            return out
        f16_ok = False
        for nm, Qx in (("float64", Q), ("float32", Q.astype(np.float32)),
                       ("float16", Q.astype(np.float16).astype(np.float64))):
            r = _ranks(Qx)
            ok = rref >= 0
            ex = int((r[ok] == rref[ok]).sum()); t10 = int(((r[ok] < 10) != (rref[ok] < 10)).sum())
            t1 = int(((r[ok] == 0) != (rref[ok] == 0)).sum())
            print(f"[verify] {nm}: {len(r):,} groups ({miss} not in csv, {len(ref):,} csv rows); rank exact "
                  f"{ex:,}/{int(ok.sum()):,}; top10 membership mismatches {t10}; top1 mismatches {t1}; "
                  f"max|drank| {int(np.abs(r[ok] - rref[ok]).max())}", flush=True)
            if nm == "float16":
                f16_ok = (t10 == 0 and t1 == 0 and miss == 0 and len(ref) == len(r))
        if a.dtype == "auto":
            dt = np.float16 if f16_ok else np.float32
            print(f"[verify] dtype auto -> {np.dtype(dt).name}", flush=True)
    Q = Q.astype(dt)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    extra = {} if a.no_atoms else {"atoms": np.asarray(A, np.float32)}
    if a.pool_sizes:
        extra["pool_req"] = np.asarray(pool_req)              # requested pool size per row, 0 = all cells
    np.savez_compressed(a.out, Q=Q, truth=truth_a, keys=np.array(keys, dtype=object),
                        ncell=np.array(ncell), labels=np.asarray(labels, dtype=object), **extra)
    print(f"[cache] {Q.shape[0]:,} queries x {Q.shape[1]} dims ({Q.dtype}), {A.shape[0]:,} atoms "
          f"({'not stored' if a.no_atoms else 'stored'}) -> {a.out}")


if __name__ == "__main__":
    main()
