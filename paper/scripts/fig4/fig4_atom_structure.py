"""Model-independent: does the PROTOTYPE geometry itself encode drug-target relationships?

No drug responses. Only the trained atom table and the drug->target annotations.

  coherence  for each multi-target drug, mean pairwise cosine among its targets' prototypes, vs
             size- and popularity-matched random gene sets (each target replaced by a random
             prototype from its own bin of mean cosine to all others)
  retrieval  given ONE target of a drug, do its nearest prototypes contain the drug's OTHER
             targets more often than matched random genes? (recall@k over held-out targets)
  moa        do targets of drugs sharing a mechanism sit closer than targets of drugs that don't?
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True); ap.add_argument("--drug-meta", required=True)
ap.add_argument("--out-prefix", required=True); ap.add_argument("--n-perm", type=int, default=500)
ap.add_argument("--n-bins", type=int, default=20)
ap.add_argument("--graph-drugs", default="Ixazomib citrate,Panobinostat",
                help="drugs whose target neighbourhood is exported as a graph (panel F)")
ap.add_argument("--graph-neighbours", type=int, default=5, help="prototype neighbours per target")
ap.add_argument("--gene-sets-json", default="", help="optional: mark complex / pathway partners"); ap.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
a = ap.parse_args()
a_out = a.out_prefix
rng = np.random.default_rng(0)
u = lambda M: M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)
b = np.load(a.bank, allow_pickle=True)
nl = int(b["n_labels"]); L = b["labels"].astype(str)[:nl]; P = u(b["P"].astype(np.float32))[:nl]
pos = {g: i for i, g in enumerate(L)}
G = P @ P.T                                          # prototype-prototype cosine
pop = (G.sum(1) - 1) / (len(G) - 1)                  # mean cosine to all others = popularity
bin_of = np.clip((pop.argsort().argsort() * a.n_bins) // len(pop), 0, a.n_bins - 1)
bins = [np.flatnonzero(bin_of == j) for j in range(a.n_bins)]
draw = lambda idx: np.array([rng.choice(bins[bin_of[i]]) for i in idx], int)

m = pd.read_csv(a.drug_meta)
sp = lambda s: [t.strip() for t in str(s).replace(",", ";").split(";") if t.strip() and t.strip() != "nan"]
tg = {r.drug: [pos[g] for g in sp(r.targets) if g in pos] for r in m.itertuples()}
moa = dict(zip(m.drug, m.moa_fine.astype(str))) if "moa_fine" in m else {}
multi = {d: t for d, t in tg.items() if len(t) >= 2}
print(f"{sum(bool(v) for v in tg.values())} drugs with >=1 target in the bank "
      f"(of {len(tg)} annotated); {len(multi)} with >=2")

# ---- coherence -------------------------------------------------------------------------------
def mean_pair(idx):
    if len(idx) < 2: return np.nan
    s = G[np.ix_(idx, idx)]
    return float((s.sum() - np.trace(s)) / (len(idx) * (len(idx) - 1)))
obs = np.array([mean_pair(t) for t in multi.values()])
null = np.array([[mean_pair(draw(t)) for t in multi.values()] for _ in range(a.n_perm)])
p = (np.sum(np.nanmean(null, 1) >= np.nanmean(obs)) + 1) / (a.n_perm + 1)
print(f"target-set coherence: observed mean pairwise cosine {np.nanmean(obs):.4f} vs matched random "
      f"{np.nanmean(null):.4f} (p={p:.4f}); drugs above their own null: "
      f"{np.mean(obs > np.nanmean(null, 0)):.2f}")

# ---- retrieval of the other targets from one target -------------------------------------------
rows = []
for d, t in multi.items():
    for h in range(len(t)):
        q = t[h]; rest = set(t) - {q}
        order = np.argsort(-G[q]); order = order[order != q]
        rk = {g: i + 1 for i, g in enumerate(order)}
        best = min(rk[g] for g in rest)
        # matched-random control: same number of held-out genes, popularity-matched
        # the matched draw can return the query gene itself, which has no rank (it was removed)
        ctrl = [min(rk.get(g, nl) for g in draw(sorted(rest))) for _ in range(20)]
        rows.append(dict(drug=d, query=L[q], n_other=len(rest), rank_other=best,
                         rank_matched_median=float(np.median(ctrl)), moa=moa.get(d, "")))
r = pd.DataFrame(rows)
print(f"\nretrieve a drug's OTHER targets from one target (n={len(r)} queries, {r.drug.nunique()} drugs):")
for k in a.ks:
    print(f"  recall@{k}: {100*(r.rank_other<=k).mean():5.1f}%   matched-random {100*(r.rank_matched_median<=k).mean():5.1f}%")
print(f"  median rank {r.rank_other.median():.0f} vs matched random {r.rank_matched_median.median():.0f} (of {nl})")
r.to_csv(f"{a.out_prefix}_retrieval.csv", index=False)

# ---- same-MoA target proximity -----------------------------------------------------------------
# Drugs that share an annotated mechanism often share the literal target list (two PARP inhibitors
# are both PARP1;PARP2), which would make the contrast trivially 1.0. Pairs with ANY shared target
# are therefore reported separately and the headline uses the disjoint pairs only.
res_moa = {}
if moa:
    ds = [d for d in tg if tg[d] and moa.get(d, "") not in ("", "nan", "unclear")]
    cent = np.stack([u(P[tg[d]].mean(0, keepdims=True))[0] for d in ds])
    C = cent @ cent.T; np.fill_diagonal(C, np.nan)
    mo = np.array([moa[d] for d in ds]); ts = [set(tg[d]) for d in ds]
    iu = np.triu_indices(len(ds), 1)
    pr = pd.DataFrame(dict(a=[ds[i] for i in iu[0]], b=[ds[j] for j in iu[1]], cos=C[iu],
                           same_moa=mo[iu[0]] == mo[iu[1]], moa_a=mo[iu[0]], moa_b=mo[iu[1]],
                           shares_target=[len(ts[i] & ts[j]) > 0 for i, j in zip(*iu)]))
    pr.to_csv(f"{a_out}_centroid_pairs.csv", index=False)
    dj = pr[~pr.shares_target]
    res_moa = dict(n_pairs=len(pr), same_all=float(pr.cos[pr.same_moa].mean()),
                   diff_all=float(pr.cos[~pr.same_moa].mean()),
                   n_same_disjoint=int(dj.same_moa.sum()),
                   same_disjoint=float(dj.cos[dj.same_moa].mean()),
                   diff_disjoint=float(dj.cos[~dj.same_moa].mean()),
                   frac_same_sharing=float(pr.shares_target[pr.same_moa].mean()))
    print(f"\ntarget-centroid cosine, same-MoA drug pairs {res_moa['same_all']:.3f} "
          f"(n={int(pr.same_moa.sum())}) vs different-MoA {res_moa['diff_all']:.3f}")
    print(f"  {100*res_moa['frac_same_sharing']:.0f}% of same-MoA pairs share a literal target; "
          f"dropping those: same {res_moa['same_disjoint']:.3f} (n={res_moa['n_same_disjoint']}) "
          f"vs different {res_moa['diff_disjoint']:.3f}")
# ---- mechanism neighbours: is a drug's nearest target-centroid a same-mechanism drug? -----------
# Absolute cosines are tiny once target-sharing pairs are dropped, so score the RANKING instead:
# of a drug's k nearest other drugs (none sharing a target with it), how many share its mechanism,
# against that drug's own base rate and against a label permutation.
res_nn = {}
if moa:
    K = 5
    sm = np.array([[mo[i] == mo[j] for j in range(len(ds))] for i in range(len(ds))])
    shr = np.array([[len(ts[i] & ts[j]) > 0 for j in range(len(ds))] for i in range(len(ds))])
    def nn_stats(lab_same):
        hit, base, keep = [], [], []
        for i in range(len(ds)):
            ok = np.flatnonzero(~shr[i] & (np.arange(len(ds)) != i))
            if len(ok) < K or lab_same[i, ok].sum() == 0:
                continue
            nn = ok[np.argsort(-C[i, ok])][:K]
            hit.append(float(lab_same[i, nn].mean())); base.append(float(lab_same[i, ok].mean()))
            keep.append(ds[i])
        return np.array(hit), np.array(base), keep
    h, bs, keep = nn_stats(sm)
    perm = []
    for _ in range(a.n_perm):
        q = rng.permutation(mo)
        perm.append(nn_stats(np.array([[q[i] == q[j] for j in range(len(ds))] for i in range(len(ds))]))[0].mean())
    perm = np.array(perm)
    pv = (np.sum(perm >= h.mean()) + 1) / (a.n_perm + 1)
    res_nn = dict(k=K, n_drugs=len(keep), hit=float(h.mean()), base=float(bs.mean()),
                  fold=float(h.mean() / max(bs.mean(), 1e-9)), perm_mean=float(perm.mean()), p=float(pv))
    pd.DataFrame(dict(drug=keep, frac_same_moa_top5=h, base_rate=bs,
                      moa=[moa[d] for d in keep])).to_csv(f"{a_out}_moa_neighbours.csv", index=False)
    np.save(f"{a_out}_moa_perm.npy", perm)
    print(f"\nmechanism neighbours (top {K} target-centroid neighbours, no shared target, "
          f"{len(keep)} drugs): {100*h.mean():.1f}% share the mechanism vs base rate "
          f"{100*bs.mean():.1f}% ({res_nn['fold']:.1f}x), permuted {100*perm.mean():.1f}%, p={pv:.4f}")

# ---- neighbourhood graph around one drug's targets (panel F) -----------------------------------
# Nodes: the drug's annotated targets plus each target's nearest prototypes. Edges: a pair is joined
# when either prototype is among the other's nearest --graph-neighbours in the whole bank, so the
# picture is the bank's own neighbour structure, not a threshold chosen per drug.
gs = {}
if a.gene_sets_json:
    sets = {k: set(v) for k, v in json.load(open(a.gene_sets_json)).items() if len(v) <= 30}
    for k, v in sets.items():
        for g in v:
            gs.setdefault(g, set()).add(k)
order_all = np.argsort(-G, axis=1)
knn = {i: [j for j in order_all[i, :a.graph_neighbours + 1] if j != i][:a.graph_neighbours]
       for i in range(len(G))}
nodes, edges = [], []
for dn in [x.strip() for x in a.graph_drugs.split(",") if x.strip()]:
    t = tg.get(dn, [])
    if not t:
        print(f"graph: {dn!r} has no target in the bank"); continue
    nb = sorted(set(t) | {j for i in t for j in knn[i]})
    kset = set().union(*[gs.get(L[i], set()) for i in t]) if gs else set()
    for i in nb:
        role = "target" if i in t else ("partner" if gs.get(L[i], set()) & kset else "other")
        nodes.append(dict(drug=dn, gene=L[i], role=role,
                          cos_to_target=float(max(G[i, j] for j in t if j != i))))
    for x_, i in enumerate(nb):
        for j in nb[x_ + 1:]:
            if j in knn[i] or i in knn[j]:
                edges.append(dict(drug=dn, a=L[i], b=L[j], cos=float(G[i, j])))
    print(f"graph {dn}: {len(t)} targets, {len(nb)} nodes, "
          f"{sum(e['drug'] == dn for e in edges)} edges, "
          f"{sum(n['role'] == 'partner' for n in nodes if n['drug'] == dn)} complex partners")
pd.DataFrame(nodes).to_csv(f"{a_out}_graph_nodes.csv", index=False)
pd.DataFrame(edges).to_csv(f"{a_out}_graph_edges.csv", index=False)

json.dump(dict(n_drugs=len(tg), n_multi=len(multi), n_queries=len(r), coherence_obs=float(np.nanmean(obs)),
               coherence_null=float(np.nanmean(null)), coherence_p=float(p),
               recall={str(k): [float((r.rank_other <= k).mean()),
                                float((r.rank_matched_median <= k).mean())] for k in a.ks},
               median_rank=[float(r.rank_other.median()), float(r.rank_matched_median.median())],
               moa=res_moa, moa_nn=res_nn),
          open(f"{a.out_prefix}_meta.json", "w"), indent=1)
