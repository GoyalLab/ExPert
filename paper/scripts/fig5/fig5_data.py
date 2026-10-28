#!/usr/bin/env python
# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Fig. 5 data: the production model read on alcohol-associated liver disease (ALD) nuclei.

No model load. Everything comes from ONE encode of fig5_variants.py (--variant-dir, e.g.
variants/v0.1.0__mito_mt): mitochondrially encoded genes + metallothioneins dropped from the input;
  cells     one predict() per cell type, that type's Normal nuclei = the input baseline
  pbulk_x   each donor x cell type split into random pools, each pool's mean raw counts encoded as one row
            (Normal pools = the input baseline), z averaged per donor x cell type.
The readout is pbulk_x within a cell type, --contrast rrep (default): within one donor set ({AC1,AH1,Nor1} or
{AC2,AH2,Nor2}, so the two donors of a condition are independent replicates), cos(condition - mean of the set's
AC/AH/Normal) over the prototypes. --contrast zrep additionally z-scores every prototype across the three
conditions (the original ALD readout) but that saturates at ~sqrt2 and leaves the top ranks tied. --contrast healthy: unit(z_donor - mean z of the Normal
donors) . P over the trained prototypes; a Normal donor is compared with the other Normal donor (healthy
donor-vs-donor, the null). Chosen over 'out' (every donor outside the condition) and 'rest' (every other donor)
because it marks the most literature biology (analysis_out/fig5_zoom/b_readout_compare.py). Loss tail = +cos
("resembles losing gene G"); gain = -cos.

    a  UMAP of the nuclei (cosine, whole-query centred cell z) + the 50 top prototypes of the pbulk_x
       profiles averaged over each condition's donors and cell types, placed with umap.transform
    b  gene sets per donor x cell type (fig5_variants score stage output, --readout): signed set z; significance
       = empirical p against ALL Normal-donor set |z| (healthy donor-to-donor variation), BH -> q
    c  each disease donor x cell type profile vs the condition x cell type own-transcript DE
    d  nuclear-transport genes: z (score standardised over all prototypes) and rank in every donor x cell
       type profile, own-transcript DE

    python paper/scripts/fig5/fig5_data.py --variant-dir <variants/v0.1.0__mito_mt> --tests <tests csv.gz> \
        --bank-npz <fullbank/bank.npz> --deg-dir <readout_deg> --reactome-gmt <gmt> --out-dir <data>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

CT = ['HSC', 'Hepatocyte', 'Kupffer Cell', 'LSEC', 'T cell']
# nuclear transport = the pore, the mRNA export machinery and the karyopherins (curated by family
# name before looking at ranks; Reactome's mature-mRNA transport set is added on top)
NT_PATTERNS = {
    'nuclear pore': r'^(NUP\d+[A-Z]?\d*|NUPL\d|POM121[A-Z]?|NDC1|SEH1L|SEC13|AAAS|TPR|RAE1|GLE1|NUP62CL)$',
    'mRNA export (TREX/TREX-2)': r'^(THOC\d|ALYREF|DDX39[AB]|SARNP|CHTOP|POLDIP3|NXF1|NXT1|GANP|MCM3AP|ENY2|'
                                 r'PCID2|SEM1|CETN[23]|ZC3H11A|UAP56|SRSF3|SRSF7|DDX19[AB]|LRPPRC)$',
    'karyopherin / Ran': r'^(KPNA\d|KPNB1|XPO\d+|XPOT|CSE1L|TNPO\d|IPO\d+|RAN|RANBP\d+|RANGAP1|RCC1|NUTF2)$',
}
REACTOME_NT = ['Transport of Mature mRNA Derived from an Intron-Containing Transcript',
               'Transport of Mature Transcript to Cytoplasm']


def ranks_desc(v):
    o = np.argsort(-v, kind='stable')
    r = np.empty(len(v), int)
    r[o] = np.arange(1, len(v) + 1)
    return r


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def read_gmt(path, universe, lo=10, hi=500):
    u, sets = set(universe), {}
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        g = sorted(set(x.split(',')[0] for x in f[2:] if x) & u)
        if lo <= len(g) <= hi:
            sets[f[0]] = g
    return sets


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def nt_family(labels, sets):
    fam = {}
    for f, pat in NT_PATTERNS.items():
        for l in labels:
            if re.match(pat, l) and l not in fam:
                fam[l] = f
    for n in REACTOME_NT:
        for l in sets.get(n, []):
            fam.setdefault(l, 'mRNA export (TREX/TREX-2)' if not l.startswith('NUP') else 'nuclear pore')
    return fam


def load_bank(path, which='trained'):
    b = np.load(path, allow_pickle=True)
    tr = b['trained'].astype(bool) if which == 'trained' else np.ones(len(b['labels']), bool)
    return [str(l) for l in b['labels'][tr]], unit(b['P'][tr].astype(np.float32)), b


def cond_of_donor(d):
    return 'Normal' if d.startswith('Nor') else d[:2]


def npz_profiles(path, key, labels):
    """{(donor, celltype): score vector} from fig5_predict.py score output (predict() readout of the rrep
    contrasts; key = <unit>_<readout>, e.g. group_deflate), and donor -> condition."""
    z = np.load(path, allow_pickle=True)
    assert [str(l) for l in z['labels']] == list(labels), f'{path}: labels differ from the bank'
    un = key.split('_')[0]
    S = z[key].astype(np.float64)
    keys = [tuple(k.split('|')) for k in z[f'{un}_keys']]
    return {k: S[i] for i, k in enumerate(keys)}, {d: cond_of_donor(d) for d, _ in keys}


def pbx_profiles(a, Pn):
    """{(donor, celltype): donor-vs-rest prototype scores} from the encoded pseudobulk pools, and donor ->
    condition. Mirrors fig5_variants.profiles()['pbulk_x'] exactly."""
    src = Path(a.variant_dir)
    obsp = pd.read_csv(src / 'obs_pbulk.csv')
    Zp = np.load(src / 'z_pbulk.npy').astype(np.float32)
    prof = {}
    if a.contrast in ('zrep', 'rrep'):
        return zrep_profiles(obsp, Zp, Pn, a, raw=a.contrast == 'rrep'), dict(zip(obsp.donor, obsp.condition))
    for t in CT:
        pt = (obsp.celltype == t).to_numpy()
        ds = sorted(obsp.donor[pt].unique())
        pm = {d: Zp[pt & (obsp.donor == d).to_numpy()].mean(0) for d in ds}
        cnd = {e: obsp.condition[obsp.donor == e].iloc[0] for e in ds}
        for d in ds:
            nor = [e for e in ds if e != d and cnd[e] == a.control_condition]
            if a.contrast == 'healthy':   # fixed healthy reference: the Normal donors (Normal: the other Normal)
                ref = nor
            elif a.contrast == 'out':     # every donor of this cell type outside d's condition (Normal: other Normal)
                ref = nor if cnd[d] == a.control_condition else [e for e in ds if cnd[e] != cnd[d]]
            else:                       # 'rest': every other donor of this cell type
                ref = [e for e in ds if e != d]
            if ref:
                prof[(d, t)] = unit(pm[d] - np.mean([pm[e] for e in ref], 0)) @ Pn.T
    return prof, dict(zip(obsp.donor, obsp.condition))


def zrep_profiles(obsp, Zp, Pn, a, raw=False):
    """The original ALD readout (plot_perturbation_enrichment score='z_score') on independent donor sets: within a
    cell type, {AC1, AH1, Nor1} and {AC2, AH2, Nor2} each give one profile per condition (cosine of its deviation from
    the set's centre), every prototype z-scored ACROSS the conditions; donor d carries its own set's vector. Mirrors
    fig5_variants profiles()['pbulk_x_zrep'] exactly (a set missing its Normal donor borrows the other one)."""
    def zacross(vecs):
        S = np.stack([unit(v - np.mean(vecs, 0)) @ Pn.T for v in vecs])
        return (S - S.mean(0)) / np.where(S.std(0) > 1e-9, S.std(0), 1.0)
    prof = {}
    for t in CT:
        pt = (obsp.celltype == t).to_numpy()
        ds = sorted(obsp.donor[pt].unique())
        cnd = {e: obsp.condition[obsp.donor == e].iloc[0] for e in ds}
        pm = {d: Zp[pt & (obsp.donor == d).to_numpy()].mean(0) for d in ds}
        conds = sorted(set(cnd.values()))
        nor = [e for e in ds if cnd[e] == a.control_condition]
        for k in ('1', '2'):
            trip = {c: [e for e in ds if e.endswith(k) and cnd[e] == c] for c in conds}
            if not trip.get(a.control_condition):
                trip[a.control_condition] = nor[:1]
            if any(not v for v in trip.values()):
                continue
            vecs = [pm[trip[c][0]] for c in conds]
            Zr = (np.stack([unit(v - np.mean(vecs, 0)) @ Pn.T for v in vecs]) if raw   # rrep: un-normalised
                  else zacross(vecs))
            for c in conds:
                if trip[c][0].endswith(k):
                    prof[(trip[c][0], t)] = Zr[conds.index(c)]
    return prof


# ------------------------------------------------------------------------------------------- a
def panel_a(a, labels, prof, cond_of, bank, out):
    import umap
    src = Path(a.variant_dir)
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    Z = np.load(src / 'z_cells.npy').astype(np.float32)
    keep = obs.done.to_numpy()
    obs, Z = obs[keep].reset_index(drop=True), Z[keep]
    Z -= Z.mean(0, keepdims=True)                                 # whole-query centre
    Zn = unit(Z)
    # condition score = mean pbulk_x profile over that condition's donors and cell types; the two lists are
    # taken rank by rank in turn until n_atoms prototypes are distinct (kept under the better-ranked arm)
    S = {c: np.mean([v for (d, _), v in prof.items() if cond_of[d] == c], 0) for c in ('AC', 'AH')}
    order = {c: np.argsort(-S[c]) for c in S}
    rows, seen, r = [], set(), 0
    while len(seen) < a.n_atoms:
        for c in ('AC', 'AH'):
            j = order[c][r]
            if labels[j] not in seen and len(seen) < a.n_atoms:
                seen.add(labels[j])
                rows.append(dict(arm=c, arm_rank=r + 1, gene=labels[j], score=float(S[c][j])))
        r += 1
    atoms = pd.DataFrame(rows)
    pos = {str(l): i for i, l in enumerate(bank['labels'])}
    Pa = unit(bank['P'][[pos[g] for g in atoms.gene]].astype(np.float32))
    print(f'[a] umap on {len(Zn):,} nuclei x {Zn.shape[1]}, {len(atoms)} prototypes', flush=True)
    um = umap.UMAP(n_neighbors=a.n_neighbors, min_dist=0.3, metric='cosine', random_state=0)
    E = um.fit_transform(Zn)
    EP = um.transform(Pa)
    cells = pd.DataFrame({'condition': obs.condition, 'celltype': obs.celltype, 'donor': obs.donor,
                          'umap1': E[:, 0], 'umap2': E[:, 1]})
    atoms['umap1'], atoms['umap2'] = EP[:, 0], EP[:, 1]
    cells.to_csv(out / 'fig5a_cells.csv.gz', index=False)
    atoms.to_csv(out / 'fig5a_prototypes.csv', index=False)


# ------------------------------------------------------------------------------------------- b
def panel_b(a, out):
    t = pd.read_csv(a.tests)
    t = t[t.readout == a.readout].copy()
    nor = t.donor.str.startswith('Nor')
    thr = float(np.quantile(np.r_[t.z[nor], -t.z[nor]], 1 - a.alpha))   # same as fig5_variants merge
    # empirical significance against healthy donor-to-donor variation: p = share of all Normal-donor set |z|
    # (every set, cell type, donor) at least as extreme; BH over the disease-donor tests (Normal rows: own BH)
    null = np.sort(np.abs(t.z[nor].to_numpy()))
    t['p_emp'] = (1 + len(null) - np.searchsorted(null, t.z.abs().to_numpy(), side='left')) / (1 + len(null))
    # loss-only reading: one-sided, against the Normal donors' LOSS z (positive tail of t.z)
    nl = np.sort(t.z[nor].to_numpy())
    t['p_emp_loss'] = (1 + len(nl) - np.searchsorted(nl, t.z.to_numpy(), side='left')) / (1 + len(nl))
    t['q_emp'] = np.nan
    for m in (~nor, nor):
        t.loc[m, 'q_emp'] = bh(t.loc[m, 'p_emp'].to_numpy())
    t['signed_z'] = -t.z                                                 # gain positive, loss negative
    sets = t.family != 'npc'
    g = t[sets].groupby(['celltype', 'donor']).z
    t.loc[sets, 'loss_rank'] = g.rank(ascending=False)
    t.loc[sets, 'gain_rank'] = g.rank(ascending=True)
    t.loc[sets, 'n_sets'] = g.transform('size')
    t['threshold'] = thr
    t.drop(columns=['readout']).to_csv(out / 'fig5b_rest.csv.gz', index=False)
    print(f'[b] {t.set.nunique()} sets x {t.donor.nunique()} donors; threshold {thr:.2f}', flush=True)


def panel_b_ora(a, out):
    """Pathways from each group's TOP-k (fig5_predict.py stage ora: hypergeometric over-representation in the top-k
    of each tail; 'condition' level = union of the condition's donors' top-k). Per level, p_emp = share of ALL
    Normal tests (both tails, every set) with p at least as small. The shown tail is the one with the smaller p;
    value = log2 fold enrichment, loss negative / gain positive (the colour convention of the z version)."""
    t = pd.read_csv(Path(a.profiles_npz).parent / f'ora__predict__{a.bank}.csv.gz')
    t = t[t.readout == a.readout].copy()
    out_ = []
    for lv, x in t.groupby('level'):
        nor = x.group.str.startswith('Nor')
        null = np.sort(x.p[nor].to_numpy())
        x = x.assign(p_emp=(1 + np.searchsorted(null, x.p.to_numpy(), side='right')) / (1 + len(null)))
        x['lfe'] = np.log2((x.hits + 0.5) / (x.expected + 0.5))
        k = ['level', 'group', 'celltype', 'family', 'set']
        w = x.pivot_table(index=k, columns='tail', values=['p', 'p_emp', 'lfe', 'hits', 'expected']).reset_index()
        w.columns = ['_'.join(c).strip('_') for c in w.columns]
        loss = w.p_loss <= w.p_gain
        w['tail'] = np.where(loss, 'loss', 'gain')
        w['signed_lfe'] = np.where(loss, -w.lfe_loss, w.lfe_gain)
        w['p_emp'] = np.where(loss, w.p_emp_loss, w.p_emp_gain)
        w['p'] = np.where(loss, w.p_loss, w.p_gain)
        out_.append(w)
    r = pd.concat(out_).rename(columns={'group': 'donor'})
    r.to_csv(out / 'fig5c_ora.csv.gz', index=False)
    print(f'[b] top-{a.top_ora} ORA ({a.readout}): {len(r):,} rows', flush=True)


EXPORT_RECEPTORS = ['NXF1', 'NXT1', 'NXT2', 'XPO1', 'XPO5', 'XPO7', 'XPOT', 'CSE1L', 'RAN', 'RANGAP1', 'RANBP1',
                    'RANBP2', 'RCC1', 'NUTF2', 'DDX19A', 'DDX19B', 'GLE1']
EXPORT_REACTOME = ['Transport of Mature Transcript to Cytoplasm',
                   'Transport of Mature mRNA Derived From an Intron-Containing Transcript']
# literature rows shown in 6b whatever their significance: (library, term prefix, label, reference)
PINS = [('Reactome', 'mRNA Splicing - Major Pathway', 'mRNA splicing', 'Chembazhi 2025'),
        ('Reactome', 'Signaling By TGFB Family Members', 'TGF-β signalling', 'Chembazhi 2025')]


def export_parts(labels, gmt, modules):
    """The grouped mRNA-export set and its parts, restricted to the bank: nuclear pore, TREX/TREX-2/export adaptors
    (readout_panels rna_export_trex), export receptors / exportins / Ran cycle, Reactome mature-transcript transport."""
    from fig5_centring_audit import NPC
    L = list(labels)
    mods = {}
    for line in open(modules):
        if line.startswith('#') or '\t' not in line:
            continue
        m, pat = line.rstrip('\n').split('\t')[:2]
        mods.setdefault(m, []).append(pat)
    ex = lambda pats: {g for p in pats for g in L if (g.startswith(p[:-1]) if p.endswith('*') else g == p)}
    rs = read_gmt(gmt, L, lo=1, hi=10_000)
    parts = {'nuclear pore': set(NPC) | ex(mods['nuclear_pore']),
             'TREX / TREX-2 / adaptors': ex(mods['rna_export_trex']),
             'export receptors / Ran': ex(EXPORT_RECEPTORS),
             'Reactome mature-mRNA transport': set().union(*[set(rs.get(n, [])) for n in EXPORT_REACTOME])}
    # 'NUP*' in the module also matches NUPR1/NUPR2 (nuclear protein 1/2, stress-response, NOT nucleoporins)
    parts = {k: sorted(g for g in v & set(L) if not g.startswith('NUPR')) for k, v in parts.items()}
    parts['mRNA export (grouped)'] = sorted(set().union(*parts.values()))
    return parts


def union_ora(a, sets, key):
    """ORA of named gene sets in each condition x cell type's union of its donors' top-k (both tails), from the
    fig5_predict.py profiles npz (profile key `key`), background = the bank."""
    from scipy.stats import hypergeom
    z = np.load(a.profiles_npz, allow_pickle=True)
    L = [str(l) for l in z['labels']]
    pos = {g: i for i, g in enumerate(L)}
    S, gk = z[key], [tuple(k.split('|')) for k in z[f'keys__{key}']]
    rows = []
    level = getattr(a, 'u_level', 'condition')
    groups = (('AC', 'AH', 'Nor') if level == 'condition' else sorted({d for d, _ in gk}))
    for t in sorted({t for _, t in gk}):
        for c in groups:
            ix = [i for i, (d, tt) in enumerate(gk) if tt == t and (d.startswith(c) if level == 'condition' else d == c)]
            for tail, sg in (('loss', 1), ('gain', -1)):
                if not ix:
                    continue
                u = set(np.concatenate([np.argsort(-sg * S[i], kind='stable')[:a.top_ora] for i in ix]).tolist())
                for name, genes in sets.items():
                    m = {pos[g] for g in genes if g in pos}
                    h = len(u & m)
                    rows.append(dict(level=level, group=c, celltype=t, tail=tail, library='curated', term=name,
                                     n_set=len(m),
                                     K=len(u), hits=h, expected=len(m) * len(u) / len(L),
                                     p=hypergeom.sf(h - 1, len(L), len(m), len(u)),
                                     members=';'.join(sorted(genes))))
    return pd.DataFrame(rows)


def panel_unbiased(a, out, per=3, max_rows=28, jac=0.5):
    """Unbiased top-k enrichment (fig5_predict.py stage unbiased: Fig. 1d libraries over each condition x cell
    type's union of its donors' top-k, per tail). p_emp vs ALL Normal-union tests; BH q over the disease tests.
    Terms: per condition x cell type x tail the `per` best with q < .05, p_emp < .01 and >= 3 hits, broad
    disease/infection terms dropped, then greedily de-duplicated (member Jaccard >= `jac` with a better term), capped
    at `max_rows`; grouped by the Fig. 1d themes (+ nuclear transport)."""
    import fig1d_plot as f1
    t = pd.read_csv(Path(a.profiles_npz).parent / f'unbiased__predict__{a.bank}.csv.gz')
    if 'readout' in t:
        t = t[t.readout == a.u_readout].copy()
    if 'level' in t:
        t = t[t.level == a.u_level].copy()
    t['block'] = 'data'
    # grouped export row, scored exactly like the library terms (and calibrated on the same Normal unions)
    parts = export_parts(np.load(a.profiles_npz, allow_pickle=True)['labels'].astype(str), a.reactome_gmt, a.modules)
    ex = union_ora(a, {'mRNA export (grouped)': parts['mRNA export (grouped)']}, a.u_readout)
    ex['block'] = 'export'
    t = pd.concat([t, ex], ignore_index=True)
    if a.u_pins == 'brows':
        # the earlier figure's literature rows (this cohort + other ALD literature), same scoring and null
        from fig5_plot import B_ROWS
        from fig5_centring_audit import NPC
        from fig5_variants import PAUSE_BRAKE
        labs = np.load(a.profiles_npz, allow_pickle=True)['labels'].astype(str)
        rs = read_gmt(a.reactome_gmt, labs, lo=1, hi=10_000)
        mods = {}
        for line in open(a.modules):
            if line.startswith('#') or '\t' not in line:
                continue
            m_, pat = line.rstrip('\n').split('\t')[:2]
            mods.setdefault(m_, []).append(pat)
        exm = lambda pats: sorted({g for p_ in pats for g in labs
                                   if (g.startswith(p_[:-1]) if p_.endswith('*') else g == p_)})
        bsets, bref = {}, {}
        for lab_, fam, st, blk, ref in B_ROWS:
            if blk == 'new':
                continue
            bsets[lab_] = {'reactome': rs.get(st, []), 'module': exm(mods.get(st, [])), 'npc': NPC,
                           'custom': PAUSE_BRAKE}[fam]
            bref[lab_] = (blk, ref)
        lit = union_ora(a, bsets, a.u_readout)
        lit['block'] = 'literature'
        lit['theme'] = lit.term.map(lambda x: 'this cohort (Chembazhi 2025)' if bref[x][0] == 'cohort'
                                    else 'other ALD literature')
        lit['reference'] = lit.term.map(lambda x: bref[x][1])
        t = pd.concat([t, lit], ignore_index=True)
    nor = t.group.str.startswith('Nor')
    null = np.sort(t.p[nor].to_numpy())
    t['p_emp'] = (1 + np.searchsorted(null, t.p.to_numpy(), side='right')) / (1 + len(null))
    t['q'] = np.nan
    t.loc[~nor, 'q'] = bh(t.p[~nor].to_numpy())
    t['lfe'] = np.log2((t.hits + 0.5) / (t.expected + 0.5))
    themes = [('nuclear transport', ('nuclear pore', 'nucleocytoplasmic', 'export from nucleus', 'nuclear export',
                                     'mrna export', 'transport of mature', 'nuclear envelope', 'import into nucleus',
                                     'nuclear transport'))] + f1.THEMES

    def theme(term):
        tl = term.lower()
        if any(b in tl for b in f1.BROAD + ('lupus', 'alcoholism', 'hcmv')):     # + KEGG disease maps
            return None
        for nm, keys in themes:
            if any(k in tl for k in keys):
                return nm
        return 'other'
    _lit_theme = t.theme.copy() if 'theme' in t else None
    t['theme'] = t.term.map(theme)
    if _lit_theme is not None:
        t.loc[t.block == 'literature', 'theme'] = _lit_theme[t.block == 'literature']
    t.loc[t.block == 'export', 'theme'] = 'nuclear transport'
    c = t[~nor & t.theme.notna() & (t.hits >= 3) & (t.q < 0.05) & (t.p_emp < 0.01) & (t.block == 'data')]
    c = c.sort_values('p').groupby(['group', 'celltype', 'tail']).head(per).sort_values('p')
    mem = dict(zip(t.library + '|' + t.term, t.members.fillna('').str.split(';').map(set)))
    kept, labs = [], set()
    for r in c.drop_duplicates(['library', 'term']).itertuples():
        k = f'{r.library}|{r.term}'
        lab = f1.short_label(r.term)
        if lab in labs or any(len(mem[k] & mem[q]) / max(1, len(mem[k] | mem[q])) >= jac for q in kept):
            continue
        kept.append(k); labs.add(lab)
        if len(kept) == max_rows:
            break
    t['key'] = t.library + '|' + t.term
    sel = t[t.key.isin(kept) & ~nor].copy()
    sel['label'] = sel.term.map(f1.short_label)
    sel['reference'] = ''
    exr = t[(t.block == 'export') & ~nor].assign(label='mRNA export (grouped)', reference='RNA FISH (this study)')
    pins = []
    if a.u_pins == 'brows':
        pins.append(t[(t.block == 'literature') & ~nor].assign(label=lambda q: q.term,
                                                                key=lambda q: 'lit|' + q.term))
    for lib, pre, lab, ref in (PINS if a.u_pins == 'pins' else []):
        x = t[(t.library == lib) & t.term.str.startswith(pre) & ~nor & (t.block == 'data')]
        x = x[x.term == x.sort_values('p').term.iloc[0]] if len(x) else x
        pins.append(x.assign(block='literature', theme='reported in ALD', label=lab, reference=ref,
                             key=lambda q: 'pin|' + q.key))
    sel = pd.concat([exr, sel] + pins, ignore_index=True)
    sel.to_csv(out / 'fig5b_unbiased.csv.gz', index=False)
    sel['level'] = a.u_level
    sel.to_csv(out / 'fig5b_unbiased.csv.gz', index=False)
    print('[u] grouped export (hepatocytes):\n' + t[(t.block == 'export') & (t.celltype == 'Hepatocyte')][
        ['group', 'tail', 'hits', 'expected', 'p', 'p_emp']].round(4).to_string(index=False), flush=True)
    print(f'[u] unbiased: {len(c)} candidate calls -> {len(kept)} terms', flush=True)


def panel_b_donor(a, out):
    """6b: per disease donor x cell type, top-200 enrichment (each donor's own top-k, profile key --u-readout) of the
    curated rows: the grouped mRNA-export set and its parts + Pol II elongation (not previously linked to ALD), the
    cohort paper's pathways and other ALD literature (fig5_plot.B_ROWS). Significance: empirical p against EVERY
    Normal-donor test (both tails) over all Reactome sets (10-500) + readout modules -- the readout grid's null."""
    from fig5_plot import B_ROWS
    labs = np.load(a.profiles_npz, allow_pickle=True)['labels'].astype(str)
    rs = read_gmt(a.reactome_gmt, labs, lo=1, hi=10_000)
    mods = {}
    for line in open(a.modules):
        if line.startswith('#') or '\t' not in line:
            continue
        m_, pat = line.rstrip('\n').split('\t')[:2]
        mods.setdefault(m_, []).append(pat)
    exm = lambda pats: sorted({g for p_ in pats for g in labs if (g.startswith(p_[:-1]) if p_.endswith('*') else g == p_)})
    parts = export_parts(labs, a.reactome_gmt, a.modules)
    rows = [('mRNA export (grouped)', parts['mRNA export (grouped)'], 'new', 'RNA FISH (this study)'),
            ('nuclear pore', parts['nuclear pore'], 'new', ''),
            ('TREX / TREX-2 / adaptors', parts['TREX / TREX-2 / adaptors'], 'new', ''),
            ('export receptors / Ran cycle', parts['export receptors / Ran'], 'new', ''),
            ('Pol II elongation', rs.get('RNA Polymerase II Transcription Elongation', []), 'new', '')]
    for lab_, fam, st, blk, ref in B_ROWS:
        if blk != 'new':
            rows.append((lab_, {'reactome': rs.get(st, []), 'module': exm(mods.get(st, []))}[fam], blk, ref))
    targets = {r[0]: r[1] for r in rows}
    null_sets = {f'r:{k}': v for k, v in read_gmt(a.reactome_gmt, labs).items()}
    null_sets.update({f'm:{k}': g for k in mods for g in [exm(mods[k])] if len(g) >= 5})
    lv = a.u_level
    a.u_level = 'donor'
    T = union_ora(a, targets, a.u_readout)
    Nl = union_ora(a, null_sets, a.u_readout)
    a.u_level = lv
    Nl = Nl[Nl.group.str.startswith('Nor')]
    null = np.sort(Nl.p.to_numpy())
    T['p_emp'] = (1 + np.searchsorted(null, T.p.to_numpy(), side='right')) / (1 + len(null))
    T['lfe'] = np.log2((T.hits + 0.5) / (T.expected + 0.5))
    w = T.pivot_table(index=['group', 'celltype', 'term'], columns='tail',
                      values=['p', 'p_emp', 'lfe', 'hits', 'expected']).reset_index()
    w.columns = ['_'.join(c).strip('_') for c in w.columns]
    loss = w.p_loss <= w.p_gain
    w['tail'] = np.where(loss, 'loss', 'gain')
    w['signed_lfe'] = np.where(loss, -w.lfe_loss, w.lfe_gain)
    w['p_emp'] = np.where(loss, w.p_emp_loss, w.p_emp_gain)
    meta = pd.DataFrame(rows, columns=['term', 'genes', 'block', 'reference']).drop(columns='genes')
    meta['order'] = np.arange(len(meta))
    w = w.merge(meta, on='term').rename(columns={'group': 'donor'})
    w.to_csv(out / 'fig5b_donor.csv.gz', index=False)
    x = w[~w.donor.str.startswith('Nor') & (w.p_emp < 0.05)]
    print(f'[b] per-donor curated rows: {len(x)} significant donor x cell type calls (null: {len(null):,} Normal tests)\n'
          + x[x.celltype == 'Hepatocyte'][['term', 'donor', 'tail', 'p_emp']].round(4).to_string(index=False),
          flush=True)


def panel_export_zoom(a, out, celltype='Hepatocyte'):
    """Export genes in one cell type, per disease donor (+ the healthy Nor1-vs-Nor2 row): loss rank and loss z
    (score standardised over the bank) under the 6b contrast; part membership for grouping."""
    z = np.load(a.profiles_npz, allow_pickle=True)
    L = np.asarray(z['labels']).astype(str)
    parts = export_parts(L, a.reactome_gmt, a.modules)
    S, gk = z[a.u_readout], [tuple(k.split('|')) for k in z[f'keys__{a.u_readout}']]
    pos = {g: i for i, g in enumerate(L)}
    order = [p for p in parts if p != 'mRNA export (grouped)']
    part_of = {}
    for p in order:
        for g in parts[p]:
            part_of.setdefault(g, p)
    rows = []
    for i, (d, t) in enumerate(gk):
        if t != celltype:
            continue
        v = S[i]
        r = ranks_desc(v)
        zz = (v - v.mean()) / v.std()
        for g, p in part_of.items():
            j = pos[g]
            rows.append(dict(donor=d, celltype=t, gene=g, part=p, loss_rank=int(r[j]), loss_z=float(zz[j]),
                             n_bank=len(L)))
    pd.DataFrame(rows).to_csv(out / 'fig5c_export_genes.csv.gz', index=False)
    print(f'[x] export zoom: {len(part_of)} genes x {len({r["donor"] for r in rows})} rows ({celltype})', flush=True)


# ------------------------------------------------------------------------------------------- c
def panel_c(a, labels, prof, cond_of, out):
    deg = pd.read_csv(Path(a.deg_dir) / 'deg.csv.gz')
    rows, stats = [], []
    for (d, t), v in sorted(prof.items()):
        cnd = cond_of[d]
        if cnd == a.control_condition:
            continue
        g = f'{d}|{t}'
        m = pd.DataFrame({'gene': labels, 'model_cos': v, 'model_loss_rank': ranks_desc(v)})
        dg = deg[deg.group == f'{cnd}|{t}'][['gene', 'score', 'logfc', 'pval_adj']]
        x = m.merge(dg, on='gene', how='left')
        x['deg_tested'] = x.score.notna()
        x['deg_sig'] = (x.pval_adj < 0.05) & (x.logfc.abs() >= a.lfc)
        tt = x[x.deg_tested].copy()
        tt['deg_abs_rank'] = ranks_desc(tt.score.abs().values)
        x = x.merge(tt[['gene', 'deg_abs_rank']], on='gene', how='left')
        x['group'] = g
        rows.append(x)
        top_m = set(x.nsmallest(a.top_n, 'model_loss_rank').gene)
        top_mt = top_m & set(tt.gene)
        sig = set(tt.gene[tt.pval_adj.lt(0.05) & tt.logfc.abs().ge(a.lfc)])
        top_d = set(tt.nsmallest(200, 'deg_abs_rank').gene)
        stats.append(dict(group=g, condition=cnd, n_bank=len(labels), n_tested=len(tt), n_sig_deg=len(sig),
                          rho_model_vs_deg=spearmanr(tt.model_cos, tt.score).statistic,
                          rho_model_vs_absdeg=spearmanr(tt.model_cos, tt.score.abs()).statistic,
                          top_n=a.top_n, top_untested=a.top_n - len(top_mt),
                          top_in_sig_deg=len(top_m & sig), top_in_sig_expected=len(top_mt) * len(sig) / len(tt),
                          top_in_top200_absdeg=len(top_m & top_d), top200_expected=len(top_mt) * 200 / len(tt)))
    pd.concat(rows).to_csv(out / 'fig5c_genes.csv.gz', index=False)
    st = pd.DataFrame(stats)
    st.to_csv(out / 'fig5c_stats.csv', index=False)
    print('[c]\n' + st.round(3).to_string(), flush=True)


# ------------------------------------------------------------------------------------------- d
def panel_d(a, labels, prof, cond_of, out):
    fam = nt_family(labels, read_gmt(a.reactome_gmt, labels, lo=1, hi=10_000))
    idx = {l: i for i, l in enumerate(labels)}
    deg = pd.read_csv(Path(a.deg_dir) / 'deg.csv.gz').set_index(['group', 'gene'])
    rows = []
    for (d, t), v in sorted(prof.items()):
        r = ranks_desc(v)
        # zrep vectors are already z across conditions (bounded at +-sqrt(2) with three groups): use them as is
        zz = v if a.contrast == 'zrep' else (v - v.mean()) / v.std()   # rrep: z over prototypes
        cnd = cond_of[d]
        for g in sorted(fam):
            dk = (f'{cnd}|{t}', g)
            rows.append(dict(donor=d, condition=cnd, celltype=t, gene=g, family=fam[g], loss_rank=int(r[idx[g]]),
                             loss_z=float(zz[idx[g]]),
                             n_bank=len(labels), deg_logfc=deg.logfc.get(dk, np.nan),
                             deg_padj=deg.pval_adj.get(dk, np.nan)))
    pd.DataFrame(rows).to_csv(out / 'fig5d_rest_genes.csv.gz', index=False)
    print(f'[d] {len(fam)} nuclear-transport genes x {len(prof)} donor x cell type profiles', flush=True)


def panel_d_dvn(a, labels, Pn, out, celltype='Hepatocyte'):
    """Hepatocytes, disease vs Normal: nuclear-transport set z of (mean z of ALL disease donors, AC + AH) minus the
    mean of the Normal donors, read like every other Fig. 5 contrast (predict()'s deflated readout at the contrast's
    depth). Per-donor contrasts (donor - Normal mean) and the healthy null (Nor1 - Nor2, both signs) alongside."""
    import expert.utils.readout as ro
    from fig5_centring_audit import pct_of, set_z
    src = Path(a.profiles_npz).parent
    e = np.load(src / 'predict_groups.npz', allow_pickle=True)
    obs = pd.read_csv(src / 'obs_groups.csv')
    m = (obs.celltype == celltype).to_numpy()
    Z, ob = e['Z'][m].astype(np.float64), obs[m].reset_index(drop=True)
    zz = dict(zip(ob.donor, Z))
    nn = dict(zip(ob.donor, ob.n_cells))
    dis = [d for d in ob.donor if not d.startswith('Nor')]
    nor = [d for d in ob.donor if d.startswith('Nor')]
    mn = np.mean([zz[d] for d in nor], 0)
    vn = sum(1 / nn[d] for d in nor) / len(nor) ** 2
    Q = [(np.mean([zz[d] for d in dis], 0) - mn, 'disease', 'AC + AH', 1 / (sum(1 / nn[d] for d in dis) / len(dis) ** 2 + vn))]
    Q += [(zz[d] - mn, 'donor', d, 1 / (1 / nn[d] + vn)) for d in dis]
    Q += [(zz[x] - zz[y], 'healthy', f'{x} - {y}', 1 / (1 / nn[x] + 1 / nn[y])) for x in nor for y in nor if x != y]
    S1 = unit(np.stack([q[0] for q in Q])) @ Pn.T
    dr = ro.deflate_rows(np.stack([q[0] for q in Q]), Pn.astype(np.float64), np.array([q[3] for q in Q]),
                         e['noise'].astype(np.float64))
    S = S1.copy()
    for j, i in enumerate(dr['rows']):
        S[i] = ro.transplant_scores(S1[i][None], dr['loss'][j][None], dr['gain'][j][None])[0]
    gmt = read_gmt(a.reactome_gmt, labels, lo=1, hi=10_000)
    pos = {l: i for i, l in enumerate(labels)}
    sets = {f: [l for l in labels if re.match(pat, l)] for f, pat in NT_PATTERNS.items()}
    sets['mRNA export to cytoplasm (Reactome)'] = [g for g in gmt.get(REACTOME_NT[1], []) if g in pos]
    names = list(sets)
    members = [np.array([pos[g] for g in sets[k]]) for k in names]
    rows = []
    for i, (_, kind, who, ne) in enumerate(Q):
        for k, (mu, z, p) in zip(names, set_z(pct_of(S[i]), members)):
            rows.append(dict(celltype=celltype, kind=kind, contrast=who, n_eff=ne, set=k, n_genes=len(sets[k]),
                             loss_z=z, p_one_sided=p))
    pd.DataFrame(rows).to_csv(out / 'fig5d_disease_vs_normal.csv', index=False)
    print('[d] disease vs Normal (hepatocytes)\n' + pd.DataFrame(rows).pivot_table(
        index='set', columns='contrast', values='loss_z').round(2).to_string(), flush=True)


# ------------------------------------------------------------------------------------------- top
def top_tables(a, prof_by_bank, cond_of, out, ks=(200,)):
    """Top-n prototypes per donor x cell type, both tails, for each bank (trained / full incl. zero-shot)."""
    deg = pd.read_csv(Path(a.deg_dir) / 'deg.csv.gz').set_index(['group', 'gene'])
    n = max(ks)
    rows = []
    for bank, (labels, zs, prof, prof_gain, prof_raw) in prof_by_bank.items():
        for (d, t), v in sorted(prof.items()):
            wg = -v if prof_gain is None else prof_gain[(d, t)]
            for tail, w in (('loss', v), ('gain', wg)):
                zz = (w - w.mean()) / w.std()
                for r, j in enumerate(np.argsort(-w)[:n]):
                    g = labels[j]
                    dk = (f'{cond_of[d]}|{t}', g)
                    rows.append(dict(bank=bank, donor=d, condition=cond_of[d], celltype=t, tail=tail, rank=r + 1,
                                     gene=g, zero_shot=bool(zs[j]), cosine=float(prof_raw[(d, t)][j]),
                                     score=float(w[j]), z=float(zz[j]),
                                     own_logfc=deg.logfc.get(dk, np.nan), own_padj=deg.pval_adj.get(dk, np.nan)))
    pd.DataFrame(rows).to_csv(out / f'fig5_top{n}_per_group.csv', index=False)
    print(f'[top] {len(rows):,} rows -> fig5_top{n}_per_group.csv (filter rank <= k for smaller k)', flush=True)
    # the original readout's display set: union of every group's top-k (loss tail), as gene x group values
    for n, (bank, (labels, zs, prof, _, _)) in [(k, bp) for k in ks for bp in prof_by_bank.items()]:
        keys = sorted(prof)
        U = sorted({labels[j] for k in keys for j in np.argsort(-prof[k])[:n]})
        idx = {l: i for i, l in enumerate(labels)}
        M = pd.DataFrame({f'{d}|{t}': [prof[(d, t)][idx[g]] for g in U] for d, t in keys}, index=U)
        M.insert(0, 'zero_shot', [bool(zs[idx[g]]) for g in U])
        M.index.name = 'gene'
        M.to_csv(out / f'fig5_union_top{n}_{bank}.csv')
        print(f'[top] union of top-{n} ({bank}): {len(U):,} perturbations', flush=True)


def hub_offsets(a, bank):
    """Stored predict() hubness offset (fig5_hubpen.py), (2, n_full) aligned to the bank, or None."""
    if not a.hub_pen:
        return None
    hz = np.load(a.hub_pen, allow_pickle=True)
    assert [str(x) for x in hz['labels']] == [str(l) for l in bank['labels']], 'hub_pen not aligned to bank'
    return hz['pen'].astype(np.float64)


def corrected(prof, pen):
    """(loss, gain) score profiles: loss cos - pen0, gain -cos - pen1 (gain None = plain mirror)."""
    if pen is None:
        return prof, None
    return ({k: v - pen[0] for k, v in prof.items()}, {k: -v - pen[1] for k, v in prof.items()})


def main(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels, Pn, bank = load_bank(a.bank_npz, a.bank)
    if a.profiles_npz:                          # predict() readout (fig5_predict.py); no hub offset on top
        prof_raw, cond_of = npz_profiles(a.profiles_npz, a.profile_key, labels)
    else:
        prof_raw, cond_of = pbx_profiles(a, Pn)
    pen_full = hub_offsets(a, bank)
    tr = bank['trained'].astype(bool)
    rows_ = tr if a.bank == 'trained' else np.ones(len(tr), bool)
    prof, prof_gain = corrected(prof_raw, None if pen_full is None else pen_full[:, rows_])
    json.dump(dict(vars(a), n_labels=len(labels), n_profiles=len(prof)), open(out / 'fig5_run.json', 'w'), indent=1)
    want = set(a.panels.split('+'))
    run = lambda p: 'all' in want or p in want
    if run('b'):
        panel_b(a, out)
        if a.profiles_npz:
            panel_b_ora(a, out)
    if run('bd') and a.profiles_npz:
        panel_b_donor(a, out)
    if run('u') and a.profiles_npz:
        panel_unbiased(a, out)
        panel_export_zoom(a, out)
    if run('c'): panel_c(a, labels, prof, cond_of, out)
    if run('d'):
        panel_d(a, labels, prof, cond_of, out)
        if a.profiles_npz:
            panel_d_dvn(a, labels, Pn, out)
    if run('a'): panel_a(a, labels, prof, cond_of, bank, out)
    if run('top'):
        full_labels = [str(l) for l in bank['labels']]
        full_P = unit(bank['P'].astype(np.float32))
        prof_full_raw = (npz_profiles(a.profiles_full_npz, a.profile_key, full_labels)[0] if a.profiles_npz
                         else pbx_profiles(a, full_P)[0])
        prof_full, prof_full_gain = corrected(prof_full_raw, pen_full)
        tl = labels if a.bank == 'trained' else [l for l, t in zip(full_labels, tr) if t]
        if a.bank == 'trained':
            ptr = (prof, prof_gain, prof_raw)
        else:
            ptr_raw = (npz_profiles(a.profiles_trained_npz, a.profile_key, tl)[0] if a.profiles_npz
                       else pbx_profiles(a, unit(bank['P'][tr].astype(np.float32)))[0])
            ptr = corrected(ptr_raw, None if pen_full is None else pen_full[:, tr]) + (ptr_raw,)
        top_tables(a, {'trained': (tl, np.zeros(len(tl), bool)) + tuple(ptr),
                       'full': (full_labels, ~tr, prof_full, prof_full_gain, prof_full_raw)}, cond_of, out,
                   ks=a.top_k)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant-dir', required=True, help='fig5_variants.py encode output (one model x mask)')
    ap.add_argument('--tests', required=True, help='fig5_variants.py tests__<model>__<mask>__trained.csv.gz')
    ap.add_argument('--bank-npz', required=True, help='that model\'s readout fullbank/bank.npz')
    ap.add_argument('--deg-dir', required=True, help='results/readout_deg (model-independent DEG tables)')
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--panels', default='all', help="'+'-list of a,b,c,d,top or 'all'")
    ap.add_argument('--control-condition', default='Normal')
    ap.add_argument('--readout', default='pbulk_x_rrep', help='readout column of --tests used for b')
    ap.add_argument('--bank', default='trained', choices=['trained', 'full'],
                    help='prototype rows for a/c/d (full = with the zero-shot f(missing) rows)')
    ap.add_argument('--profiles-npz', default='', help='fig5_predict.py profiles__predict__<bank>.npz for --bank; '
                                                      'replaces the pbx_profiles recomputation')
    ap.add_argument('--profiles-trained-npz', default='', help='top: the trained-bank npz (when --bank full)')
    ap.add_argument('--profiles-full-npz', default='', help='top: the full-bank npz')
    ap.add_argument('--profile-key', default='group_deflate', help='<unit>_<readout> in the npz')
    ap.add_argument('--top-ora', type=int, default=200, help='top-k per donor for the union enrichment')
    ap.add_argument('--u-readout', default='group_deflate__ctrl',
                    help='profile key behind 6b / 6c (default: each donor vs the Normal donors of its cell type)')
    ap.add_argument('--u-level', default='condition', choices=['condition', 'donor'],
                    help='6b columns: condition (union of its donors\' top-k) or each donor\'s own top-k')
    ap.add_argument('--u-pins', default='pins', choices=['pins', 'brows', 'none'],
                    help='literature block: pins (splicing, TGF-b) or brows (the earlier figure\'s cohort + '
                         'literature rows)')
    ap.add_argument('--modules', default='', help='readout_panels.tsv (export parts: nuclear_pore, rna_export_trex)')
    ap.add_argument('--hub-pen', default='', help='stored predict() hubness offset (fig5_hubpen.py): a/c/d/top rank '
                                                  'on cos - pen0 (loss) / -cos - pen1 (gain)')
    ap.add_argument('--contrast', default='rrep', choices=['rrep', 'zrep', 'healthy', 'out', 'rest'],
                    help='a/c/d/top profile: rrep (default) = condition vs its donor set\'s AC/AH/Normal mean, '
                         'un-normalised; zrep = the same z-scored per prototype (saturates at ~sqrt2)')
    ap.add_argument('--alpha', type=float, default=0.01, help='Normal donors exceed the b threshold this often')
    ap.add_argument('--top-n', type=int, default=50, help='model top-n for the DEG overlap counts')
    ap.add_argument('--top-k', type=int, nargs='+', default=[100, 200, 500],
                    help='top-k per group for the union tables (original: 200); the per-group table keeps max(k)')
    ap.add_argument('--lfc', type=float, default=0.25, help='|logFC| floor for a significant DEG')
    ap.add_argument('--n-atoms', type=int, default=50, help='distinct top prototypes over both conditions (a)')
    ap.add_argument('--n-neighbors', type=int, default=30)
    main(ap.parse_args())
