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

"""Fig. 5 (final): ExPert v0.1.0 on alcohol-associated liver disease nuclei, ONE readout configuration everywhere --
the winner of the readout chain (fig5_readout_chain.py, data/readout_chain/chain_scorecard.csv):

    input    one pseudobulk per donor x cell type (mean raw counts, mito_mt mask), encoded by ONE predict() call with
             center_by = cell type and the Normal donors' pools as the input baseline (fig5_predict.py encode)
    latent   each cell type's mean pseudobulk latent subtracted (unweighted over donors) -- removes the offset that
             made every absolute read look like healthy (disease vs Normal reads r .95-1.00 without it)
    groups   condition x cell type (all donors of a condition pooled, nucleus-weighted); donor x cell type for the
             replication supplement
    readout  cosine to the TRAINED prototype bank; top-200 per group and tail (loss = resembles losing the gene,
             gain = the opposite) -> over-representation of pathways (hypergeometric, background = the bank)

stage data   tables for every panel (+ the UMAP), no model load
stage plot   panels (pdf + svg + png 300 dpi):
    fig5b_umap                    nuclei UMAP, healthy vs ALD, top prototypes placed with umap.transform
    fig5c_orthogonality           ExPert rank vs own-transcript DE rank, per hepatocyte group, + Spearman
    fig5d_volcano                 own-transcript DE volcano, B's labelled genes marked
    fig5e_pathways                ExPert pathway enrichment (rows deduplicated by hit overlap)
    fig5f_pathways_deg            the same from the DEGs
    fig5l_export_families         mRNA export families, healthy vs ALD
    (G, I-K, M: fig5_bulk.py, fig5i_fish.py, fig5j_nuclei.py, fig5m_psi.py)

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_final.py --stage data|plot ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

CT = ['HSC', 'Hepatocyte', 'Kupffer Cell', 'LSEC', 'T cell']
SRC = None  # source-table dir (--src-dir); stage_plot, or any caller of the panel functions, sets it
HIGHLIGHT = [('mRNA export', 'mRNA export (grouped)'), ('mRNA splicing', 'mRNA splicing')]
# literature rows (fig5_plot.B_ROWS cohort + other ALD literature); shown in 6b when enriched in ALD (p < .05)
LIT = [('mRNA splicing', None), ('adult hepatocyte TFs', 'module: adult_hep_tf'),
       ('WNT / TCF signalling', 'Signaling by WNT'), ('Hippo signalling', 'Signaling by Hippo'),
       ('TGF-β signalling', 'Signaling by TGFB Family Members'), ('IL-1 signalling', 'Interleukin-1 Signaling'),
       ('TNFR1 → NF-κB', 'TNFR1-induced NF-kappa-B Signaling Pathway'),
       ('PLK1 / mitotic entry', 'Polo-like Kinase Mediated Events'),
       ('cytosolic translation', 'Cap-dependent Translation Initiation'),
       ('mitochondrial translation', 'Mitochondrial Translation'), ('SUMOylation', 'SUMOylation'),
       ('chromatin modification', 'Chromatin Modifying Enzymes'), ('macroautophagy', 'Macroautophagy'),
       ('amino acids → mTORC1', 'Amino Acids Regulate mTORC1'), ('proteasome assembly', 'Proteasome Assembly')]
REF = {'adult hepatocyte TFs': 'Chembazhi 2025', 'WNT / TCF signalling': 'Chembazhi 2025',
       'Hippo signalling': 'Chembazhi 2025', 'TGF-β signalling': 'Chembazhi 2025', 'IL-1 signalling': 'Chembazhi 2025',
       'TNFR1 → NF-κB': 'Chembazhi 2025', 'PLK1 / mitotic entry': 'Chembazhi 2025',
       'cytosolic translation': 'Rothschild 1971', 'mitochondrial translation': 'Hao 2020',
       'SUMOylation': 'Chandla 2026', 'chromatin modification': 'Ghare 2023', 'macroautophagy': 'Chao 2018',
       'amino acids → mTORC1': 'Davuluri 2021', 'proteasome assembly': 'Wang 2016',
       'mRNA splicing': 'Chembazhi 2025', 'mRNA export': 'RNA FISH (this study)'}


def cond_of(d):
    return 'Normal' if d.startswith('Nor') else d[:2]


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


# ------------------------------------------------------------------------------------------- data
def profiles(a):
    """Winning configuration: condition- and donor-level cosine profiles over the trained bank."""
    e = np.load(Path(a.predict_dir) / 'predict_groups.npz', allow_pickle=True)
    obs = pd.read_csv(Path(a.predict_dir) / 'obs_groups.csv')
    obs['condition'] = obs.donor.map(cond_of)
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = [str(x) for x in b['labels'][tr]]
    Pn = unit(b['P'][tr].astype(np.float64))
    Z = e['Z'].astype(np.float64)
    for c in obs.celltype.unique():                      # latent: cell type's mean over its donors (unweighted)
        m = np.flatnonzero((obs.celltype == c).to_numpy())
        Z[m] -= Z[m].mean(0)
    w = obs.n_cells.to_numpy(np.float64)
    cond = {f'{cd}|{c}': np.average(Z[ix], axis=0, weights=w[ix])
            for (cd, c), ix in obs.groupby(['condition', 'celltype']).indices.items()}
    don = {f'{d}|{c}': Z[i] for i, (d, c) in enumerate(zip(obs.donor, obs.celltype))}
    Sc = {k: unit(v) @ Pn.T for k, v in cond.items()}
    Sd = {k: unit(v) @ Pn.T for k, v in don.items()}
    return labels, Pn, Sc, Sd, b


def gene_sets(a, labels):
    import fig5_rank as fr
    from fig5_data import read_gmt
    sets = fr.gene_sets(a, labels)                        # Reactome 10-500 + modules + export + splicing (incl. SR)
    allr = read_gmt(a.reactome_gmt, labels, lo=1, hi=10_000)
    for lab, key in LIT:
        if key and not key.startswith('module: '):
            sets[f'lit: {lab}'] = allr.get(key, [])
        elif key:
            sets[f'lit: {lab}'] = sets.get(key, [])
    sets['lit: mRNA splicing'] = sets['mRNA splicing']
    return {k: v for k, v in sets.items() if len(v) >= 5}


def ora_table(S, columns, sets, labels, top):
    """columns: {name: [profile keys]} -> long table (column, set, tail, hits, expected, p) of the union top-k."""
    pos = {g: i for i, g in enumerate(labels)}
    names = list(sets)
    M = np.zeros((len(names), len(labels)), np.float32)
    for i, k in enumerate(names):
        M[i, [pos[g] for g in sets[k] if g in pos]] = 1
    nset = M.sum(1)
    rows = []
    for cn, keys in columns.items():
        keys = [k for k in keys if k in S]
        if not keys:
            continue
        for tail, sg in (('loss', 1), ('gain', -1)):
            u = sorted(set().union(*[set(np.argsort(-sg * S[k], kind='stable')[:top].tolist()) for k in keys]))
            v = np.zeros(len(labels), np.float32)
            v[u] = 1
            h = M @ v
            rows.append(pd.DataFrame(dict(column=cn, set=names, tail=tail, hits=h.astype(int), K=len(u),
                                          expected=nset * len(u) / len(labels),
                                          p=hypergeom.sf(h - 1, len(labels), nset, len(u)))))
    return pd.concat(rows, ignore_index=True)


UMAP_CENTRING = {
    'scoring': "each cell type's mean latent (equal weight per donor) subtracted -- the centring the readout uses",
    'control': "each cell type's Normal mean latent subtracted (centred on control in z)",
    'query': 'the whole query mean subtracted (one shift for every nucleus)'}


def umap_panels(a, out, labels, Pn, Sc):
    """Nuclei (input centred per cell type on Normal nuclei, fig5_readout_chain encode ct_normal) under each latent
    centring, the top-n_atoms AC / AH prototypes placed with umap.transform."""
    import umap
    ec = np.load(Path(a.cells_dir) / 'enc__ct_normal.npz')
    oc = pd.read_csv(Path(a.cells_dir) / 'obs__ct_normal.csv.gz')
    S = {cd: np.mean([Sc[f'{cd}|{c}'] for c in CT if f'{cd}|{c}' in Sc], 0) for cd in ('AC', 'AH')}
    order = {cd: np.argsort(-S[cd]) for cd in S}
    prow, seen, r = [], set(), 0
    while len(seen) < a.n_atoms:
        for cd in ('AC', 'AH'):
            j = order[cd][r]
            if labels[j] not in seen and len(seen) < a.n_atoms:
                seen.add(labels[j])
                prow.append(dict(arm=cd, arm_rank=r + 1, gene=labels[j], score=float(S[cd][j]), idx=int(j)))
        r += 1
    for var in a.umap_centring:
        Zc = ec['Z'].astype(np.float32)
        if var == 'query':
            Zc -= Zc.mean(0, keepdims=True)
        else:
            for c, ix in oc.groupby('celltype').indices.items():
                sub = oc.iloc[ix]
                ref = sub if var == 'scoring' else sub[sub.condition == 'Normal']
                centre = np.mean([Zc[ix[(sub.donor == d_).to_numpy()]].mean(0) for d_ in ref.donor.unique()], 0)
                Zc[ix] -= centre
        pro = pd.DataFrame(prow)
        um = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=0)
        E = um.fit_transform(unit(Zc))
        EP = um.transform(Pn[pro.idx.to_numpy()].astype(np.float32))
        cells = oc.assign(umap1=E[:, 0], umap2=E[:, 1])
        cells.to_csv(out / f'fig5a_cells_{var}.csv.gz', index=False)
        pro[['umap1', 'umap2']] = EP
        pro.drop(columns='idx').to_csv(out / f'fig5a_prototypes_{var}.csv', index=False)
        print('umap', var, flush=True)


def deg_ora(a, out, celltype='Hepatocyte'):
    """Top-200 DOWN ('loss') and UP ('gain') DE genes per condition (DE statistic vs Normal: the pseudobulk Wald stat)
    -> the same ORA; background = every gene tested in that cell type; columns AC, AH and ALD (union)."""
    import fig5_rank as fr
    deg = pd.read_csv(a.deg)
    H = deg[deg.group.isin([f'AC|{celltype}', f'AH|{celltype}'])]
    G = sorted(set(H.gene))
    sets = {k: v for k, v in fr.gene_sets(a, G).items() if k.startswith('Reactome: ') or k in dict(HIGHLIGHT).values()}
    pos = {g: i for i, g in enumerate(G)}
    rows = []
    cols = {'AC': [f'AC|{celltype}'], 'AH': [f'AH|{celltype}'], 'ALD': [f'AC|{celltype}', f'AH|{celltype}']}
    for cn, groups in cols.items():
        for tail, asc in (('loss', True), ('gain', False)):
            u = set()
            for g in groups:
                u |= set(H[H.group == g].sort_values('score', ascending=asc).gene.head(a.top))
            for k, v in sets.items():
                m = set(v)
                h = len(u & m)
                rows.append(dict(column=cn, set=k, tail=tail, hits=h, K=len(u), expected=len(m) * len(u) / len(G),
                                 p=hypergeom.sf(h - 1, len(G), len(m), len(u)), members=';'.join(sorted(m))))
    pd.DataFrame(rows).to_csv(out / 'fig5b_ora_deg.csv.gz', index=False)


def stage_data(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels, Pn, Sc, Sd, bank = profiles(a)
    if a.umap_only:
        umap_panels(a, out, labels, Pn, Sc)
        return
    sets = gene_sets(a, labels)
    np.savez_compressed(out / 'fig5_profiles.npz', labels=np.asarray(labels),
                        cond_keys=np.asarray(list(Sc)), cond=np.stack(list(Sc.values())).astype(np.float32),
                        donor_keys=np.asarray(list(Sd)), donor=np.stack(list(Sd.values())).astype(np.float32))
    # ---- 6b: hepatocytes, ALD / AC / AH / healthy; supplement: every cell type, every donor
    cols = {'ALD': ['AC|Hepatocyte', 'AH|Hepatocyte'], 'AC': ['AC|Hepatocyte'], 'AH': ['AH|Hepatocyte'],
            'healthy': ['Normal|Hepatocyte']}
    for c in CT:
        for cd in ('AC', 'AH', 'Normal'):
            cols[f'{cd}|{c}'] = [f'{cd}|{c}']
    T = ora_table(Sc, cols, sets, labels, a.top)
    T.to_csv(out / 'fig5b_ora.csv.gz', index=False)
    D = ora_table(Sd, {k: [k] for k in Sd if k.endswith('|Hepatocyte')}, sets, labels, a.top)
    D.to_csv(out / 'fig5b_ora_donors.csv.gz', index=False)
    # ---- 6c: hepatocyte export genes
    from fig5_data import export_parts
    parts = export_parts(labels, a.reactome_gmt, a.modules)
    part_of = {}
    for p_ in ('nuclear pore', 'TREX / TREX-2 / adaptors', 'export receptors / Ran', 'Reactome mature-mRNA transport'):
        for g in parts[p_]:
            part_of.setdefault(g, p_)
    pos = {g: i for i, g in enumerate(labels)}
    rows = []
    for cd in ('AC', 'AH', 'Normal'):
        v = Sc[f'{cd}|Hepatocyte']
        r = np.empty(len(v), int)
        r[np.argsort(-v, kind='stable')] = np.arange(1, len(v) + 1)
        zz = (v - v.mean()) / v.std()
        rows += [dict(condition=cd, gene=g, part=p_, loss_rank=int(r[pos[g]]), loss_z=float(zz[pos[g]]))
                 for g, p_ in part_of.items()]
    pd.DataFrame(rows).to_csv(out / 'fig5c_export_genes.csv', index=False)
    # ---- 6d: orthogonality to own-transcript DE (condition x cell type)
    deg = pd.read_csv(a.deg)
    grows, stats = [], []
    for k, v in Sc.items():
        cd, c = k.split('|')
        if cd == 'Normal':
            continue
        r = np.empty(len(v), int)
        r[np.argsort(-v, kind='stable')] = np.arange(1, len(v) + 1)
        m = pd.DataFrame({'gene': labels, 'score': v, 'model_rank': r})
        dg = deg[deg.group == k][['gene', 'score', 'logfc', 'pval_adj']].rename(columns={'score': 'deg_score'})
        x = m.merge(dg, on='gene')
        x['deg_rank'] = x.deg_score.abs().rank(ascending=False, method='first').astype(int)
        x['deg_sig'] = (x.pval_adj < 0.05) & (x.logfc.abs() >= a.deg_lfc)
        x['group'] = k
        grows.append(x)
        top = set(x.nsmallest(a.top_n, 'model_rank').gene)
        sig = set(x.gene[x.deg_sig])
        stats.append(dict(group=k, condition=cd, celltype=c, n_tested=len(x), n_sig_deg=len(sig),
                          rho=spearmanr(x.score, x.deg_score).statistic, top_n=a.top_n,
                          top_in_sig_deg=len(top & sig), expected=len(top) * len(sig) / len(x)))
    pd.concat(grows).to_csv(out / 'fig5d_genes.csv.gz', index=False)
    pd.DataFrame(stats).to_csv(out / 'fig5d_stats.csv', index=False)
    # ---- 6b (DE): the same top-200 ORA on the own-transcript DE lists (hepatocytes; down = 'loss', up = 'gain')
    deg_ora(a, out)
    # ---- 6a: UMAP of the nuclei + top prototypes per condition (profiles averaged over cell types)
    if not a.skip_umap:
        umap_panels(a, out, labels, Pn, Sc)
    print('data written to', out, flush=True)


# ------------------------------------------------------------------------------------------- plot
EXPERT = '#E69F00'                  # fig2_common.EXPERT_COLOR, paper-wide


def best_tail_rows(T, ref_col, n_more_p=1e-3, lit_p=0.05):
    """Rows of 6b: highlights, then literature rows enriched in the reference column (p < lit_p); every row in the
    reference column's stronger tail. Returns rows [(label, set, block)], the tail per set and N more."""
    R = T[T.column == ref_col]
    bt = R.loc[R.groupby('set').p.idxmin()].set_index('set')
    rows = [(lab, key, 'highlight') for lab, key in HIGHLIGHT]
    lit = [(lab, f'lit: {lab}') for lab, _ in LIT if lab != 'mRNA splicing' and f'lit: {lab}' in bt.index]
    lit = sorted([x for x in lit if bt.loc[x[1], 'p'] < lit_p], key=lambda x: bt.loc[x[1], 'p'])
    rows += [(lab, key, 'literature') for lab, key in lit]
    hv = T[T.column == 'healthy'].set_index(['set', 'tail']).p
    shown = {k for _, k, _ in rows} | {f'lit: {lab}' for lab, _ in LIT} | {k for _, k in HIGHLIGHT}
    n_more = sum(1 for s_, r_ in bt.iterrows() if s_ not in shown and r_.p < n_more_p
                 and hv.get((s_, r_['tail']), 1.0) > 0.05)
    return rows, bt['tail'], n_more


BAD_TERMS = ('hiv', 'hcmv', 'influenza', 'viral', 'virus', 'sars', 'hepatitis', 'carcinoma', 'cancer', 'disease',
             'defective', 'infection', 'susceptib')
# hand-shortened labels for Reactome names too long for a row (instead of truncating with an ellipsis)
SHORT = {k.lower(): v for k, v in {
    'Reactome: SUMOylation of DNA Damage Response and Repair Proteins': 'SUMOylation of DNA repair proteins',
    'Reactome: NR1H3 & NR1H2 Regulate Gene Expression Linked to Cholesterol Transport and Efflux':
        'LXR-regulated cholesterol efflux',
    'Reactome: Regulation of HSF1-mediated Heat Shock Response': 'HSF1 heat shock response',
    'Reactome: RUNX1 Interacts With Co-Factors Whose Precise Effect on RUNX1 Targets Is Not Known':
        'RUNX1 co-factor interactions',
    'Reactome: Regulation of IGF Transport and Uptake by Insulin-like Growth Factor Binding Proteins (IGFBPs)':
        'IGF transport and uptake by IGFBPs',
    'Reactome: Binding and Uptake of Ligands by Scavenger Receptors': 'ligand uptake by scavenger receptors',
    'Reactome: FASTK Family Proteins Regulate Processing and Stability of Mitochondrial RNAs':
        'FASTK regulation of mitochondrial RNAs',
    'Reactome: Activation of the AP-1 Family of Transcription Factors': 'AP-1 transcription factor activation',
    'Reactome: B-WICH Complex Positively Regulates rRNA Expression': 'B-WICH activation of rRNA expression',
    'Reactome: NOTCH1 Intracellular Domain Regulates Transcription': 'NOTCH1 (NICD1) transcription',
}.items()}


def top_rows(T, col, members, n=20, n_more_p=1e-3, highlight=True, hits=None, folded=None, order='p'):
    """Unbiased rows: Reactome pathways + the grouped mRNA export / splicing sets ranked by the column's best-tail p;
    viral / disease-named terms dropped; a set overlapping an already chosen one by >= 50% of the smaller set is
    redundant (export / splicing are chosen first so they keep their own rows). Export and splicing are always shown
    (highlight=True) or left out entirely (highlight=False: the top n other pathways only). hits ({set: genes that
    actually hit, in the set's best tail}) switches the redundancy test from SET overlap to HIT overlap, so rows carried
    by the same genes (e.g. nuclear-pore genes in glycolysis / HSF1 / SUMOylation) fold into the row chosen first;
    folded (a dict) collects {folded set: the row it folded into}.
    Returns rows [(label, set, block)] sorted by p, the tail per set and the number of further deduplicated
    pathways with p < n_more_p."""
    import fig1d_plot as f1
    R_ = T[(T.column == col) & T.set.isin(members)]
    bt = R_.loc[R_.groupby('set').p.idxmin()].sort_values('p').set_index('set')
    drop = [k for _, k in HIGHLIGHT]
    hl = drop if highlight else []
    chosen = list(hl)
    rest = []
    for k in bt.index:
        if k in drop or any(b in k.lower() for b in BAD_TERMS):
            continue
        S_ = hits if hits is not None else members
        m = S_[k]
        ov = [(len(m & S_[o]) / max(1, min(len(m), len(S_[o]))), o) for o in chosen + rest]
        if all(v < 0.5 for v, _ in ov):
            rest.append(k)
        elif folded is not None and bt.loc[k, 'p'] < n_more_p:
            folded[k] = max(ov)[1]
    shown = rest[:n - len(hl)]
    fe = np.log2((bt.hits + 0.5) / (bt.expected + 0.5))
    keys = sorted(set(shown) | set(hl), key=lambda k: bt.loc[k, 'p'] if order == 'p' else -fe[k])   # order='fe': top enrichment first
    n_more = sum(1 for k in rest if k not in keys and bt.loc[k, 'p'] < n_more_p)
    lab_of = dict((k, l) for l, k in HIGHLIGHT)

    def lab(k):
        if k in lab_of:
            return lab_of[k]
        if k.lower() in SHORT:
            return SHORT[k.lower()]
        t_ = f1.sentence_case(k.replace('Reactome: ', ''))
        return t_ if len(t_) <= 46 else t_[:44].rstrip() + '…'
    return [(lab(k), k, 'highlight' if k in lab_of else 'data') for k in keys], bt['tail'], n_more


def save(VS, fig, out, name):
    """pdf + svg + png (300 dpi) -> out"""
    import matplotlib.pyplot as plt
    out = Path(out)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{name}.svg')
    plt.close(fig)
    print('wrote', name, flush=True)


P_BINS6 = [(1e-6, 'p < 1e-6', 330), (1e-3, 'p < 0.001', 230), (0.05, 'p < 0.05', 140), (1.01, 'n.s.', 45)]


def psize(p):
    return next(s_ for thr, _, s_ in P_BINS6 if p < thr)


MARK_LABEL = {'cohort': 'reported in this cohort (Chembazhi 2025)', 'known': 'known in ALD from non-DE experiments'}


ALD_C = '#b2182b'   # ALD (AC + AH pooled); AC = lighter pink, AH = darker brown of the same red (lightness-ordered,
AC_C, AH_C = '#e57ea8', '#7a2f72'   # AC pink, AH plum; validated vs ExPert + ALD: CVD >= 12.7, normal >= 15.4)


LOSS_C, GAIN_C = '#6baed6', '#4d9221'   # 6c / 6e: down light blue, up green (CVD dE 21.7)


def marker_title(VS, ax, txt, mk, tc, y=1.02):
    """Title = [marker] text on one line, centred over the axes (marker as in 6b: ExPert diamond / DEG dot)."""
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from matplotlib.lines import Line2D
    da = DrawingArea(16, 16, 0, 0)
    da.add_artist(Line2D([8], [8], marker=mk, ms=11, ls='', mfc=tc, mec=VS.INK if mk == 'D' else tc, mew=1.0))
    ta = TextArea(txt, textprops=dict(color=tc, fontweight='bold', fontsize=VS.FONT_MIN * 1.15))
    box = HPacker(children=[da, ta], align='center', pad=0, sep=6)
    ax.add_artist(AnchoredOffsetbox(loc='lower center', child=box, pad=0, frameon=False, bbox_to_anchor=(0.5, y),
                                    bbox_transform=ax.transAxes, borderpad=0))


def column_dots(VS, cols, rows, n_more, out, name, marks=None, more_label='more pathways (p < 0.05)', title=None):
    """Pathway x condition dot plot: one column per (label, ORA table, column); colour = signed log2 fold enrichment of
    each column's stronger tail (loss / down < 0 pink, gain / up > 0 green, white at 0), size = p (6b bins).
    Optional evidence column."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    labs = [r[0] for r in rows]
    nr, nc = len(labs), len(cols)
    cell, src = {}, []
    for j, (clab, T, col) in enumerate(cols):
        R_ = T[T.column == col].set_index(['set', 'tail'])
        for i, (lab, key, blk) in enumerate(rows):
            best = min(('loss', 'gain'), key=lambda t: R_.loc[(key, t), 'p'] if (key, t) in R_.index else 1.0)
            r = R_.loc[(key, best)] if (key, best) in R_.index else pd.Series(dict(hits=0, expected=0.0, p=1.0))
            fe = max(np.log2((r.hits + 0.5) / (r.expected + 0.5)), 0.0)
            cell[i, j] = (-fe if best == 'loss' else fe, psize(r.p))
            src.append(dict(column=clab, row=lab, set=key, tail=best, hits=int(r.hits), expected=r.expected, p=r.p,
                            log2_fe=fe))
    lim = np.ceil(max(abs(v[0]) for v in cell.values()) * 2) / 2
    cmap = LinearSegmentedColormap.from_list('lossgain', [LOSS_C, '#ffffff', GAIN_C])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    fig, ax = plt.subplots(figsize=(1.0 * nc + 0.5, 0.5 * (nr + 1) + 1.0))
    ij = list(cell)
    ax.scatter([j for _, j in ij], [i for i, _ in ij], s=[cell[k][1] for k in ij], c=[cell[k][0] for k in ij],
               cmap=cmap, norm=norm, edgecolor=VS.INK2, lw=0.8, zorder=3)
    ax.set_xlim(-0.55, nc - 0.45)
    ax.set_xticks(range(nc)); ax.set_xticklabels([c for c, _, _ in cols])
    ax.xaxis.tick_top()
    if title:                                   # (text, marker, colour): e.g. ExPert + the 6b prototype diamond
        marker_title(VS, ax, *title, y=1.0 + 0.6 / fig.get_size_inches()[1])
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', color=VS.GRID, lw=0.8)
    for sp_ in ('left', 'right', 'top', 'bottom'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(length=0)
    ax.set_yticks(range(nr + 1))
    ax.set_yticklabels(labs + [f'+{n_more:,} {more_label}'])
    yl = ax.get_yticklabels()
    for tl, (_, key, blk) in zip(yl, rows):
        if blk == 'highlight':
            tl.set_fontweight('bold'); tl.set_color(VS.INK)
    yl[-1].set_fontstyle('italic'); yl[-1].set_color(VS.INK2)
    ax.set_ylim(nr + 0.5, -0.6)
    hm = []
    if marks:
        sty = {'cohort': dict(mfc=VS.INK, mec=VS.INK), 'known': dict(mfc=VS.SURFACE, mec=VS.INK)}
        for i, (_, key, _) in enumerate(rows):
            if marks.get(key) in sty:
                ax.plot([nc - 0.2], [i], marker='D', ms=9, mew=1.4, clip_on=False, ls='', zorder=4, **sty[marks[key]])
        shown_ = {marks.get(key) for _, key, _ in rows}                 # legend only for markers drawn here
        hm = [Line2D([], [], ls='', marker='D', ms=9, mew=1.4, label=l, **sty[k]) for k, l in MARK_LABEL.items()
              if k in shown_]
        ax.set_xlim(-0.55, nc - 0.05)
    w_in = fig.get_size_inches()[0]
    cax = ax.inset_axes([1.0 + 0.9 / w_in, 0.6, 0.22 / w_in, 0.34])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.outline.set_edgecolor(VS.INK2); cb.outline.set_linewidth(0.8)
    tk = [t for t in range(-int(lim), int(lim) + 1)]
    cb.set_ticks(tk); cb.set_ticklabels([f'{t:g}'.replace('-', '−') for t in tk])
    cb.set_label('log2 fold enrichment', color=VS.INK2)
    cax.text(0.5, 1.03, 'gain / up', transform=cax.transAxes, ha='center', va='bottom', color=VS.INK2)
    cax.text(0.5, -0.03, 'loss / down', transform=cax.transAxes, ha='center', va='top', color=VS.INK2)
    h = [Line2D([], [], ls='', marker='o', ms=np.sqrt(s_), mfc=VS.MUTED, mec='none', label=l) for _, l, s_ in P_BINS6]
    ax.legend(handles=h + hm, loc='upper left', bbox_to_anchor=(1.0 + 0.5 / w_in, 0.44), handletextpad=0.4,
              labelspacing=0.8, borderaxespad=0, frameon=False)
    save(VS, fig, out, name)
    pd.DataFrame(src).to_csv(SRC / f'{name}_source.csv', index=False)


def orth_stacked(VS, d, out, a, gs, cc, name='fig5c_orthogonality', top_k=500, mark=100):
    """5c (figS5_orthogonality style, stacked AC / AH at 5d's width): log10 ranks, 1 = top right, dashed diagonal.
    Only ExPert's top-top_k predictions are shown (grey), each at its snRNA DE rank (|Wald|) among ALL scored genes;
    ExPert's top-mark in ExPert orange, and any shown gene that is also a top-mark DEG in black. Spearman rho of
    ExPert vs DE rank over the shown genes in the title."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    gd = pd.read_csv(d / 'fig5d_genes.csv.gz')
    fig, axs = plt.subplots(len(a.d_groups), 1, figsize=(7.4, 6.2 * len(a.d_groups)), gridspec_kw={'hspace': 0.38})
    src = []
    for ax, grp in zip(np.atleast_1d(axs), a.d_groups):
        x = gd[gd.group == grp].copy()
        x['ex_rank'] = x.score.rank(ascending=False, method='first')
        x['de_rank'] = x.deg_score.abs().rank(ascending=False, method='first')
        n = len(x)
        top_deg = set(x.nsmallest(mark, 'de_rank').gene)
        sh = x[x.ex_rank <= top_k].copy()
        lx, ly = np.log10(sh.de_rank), np.log10(sh.ex_rank)
        ex_m, de_m = sh.ex_rank <= mark, sh.gene.isin(top_deg)
        ax.scatter(lx[~ex_m & ~de_m], ly[~ex_m & ~de_m], s=10, color=VS.MUTED, lw=0, zorder=2)
        ax.scatter(lx[ex_m], ly[ex_m], s=30, color=EXPERT, edgecolor=VS.INK, lw=0.6, zorder=3)
        ax.scatter(lx[de_m], ly[de_m], s=34, color=VS.INK, edgecolor=VS.SURFACE, lw=0.6, zorder=4)
        lim = np.log10(n * 1.4)
        ax.plot([-0.1, lim], [-0.1, lim], ls='--', color=VS.MUTED, lw=1.2, zorder=1)
        tk = [t for t in (1, 10, 100, 1000, 10000) if t <= n * 1.4]
        ax.set_xticks(np.log10(tk)); ax.set_xticklabels([f'{t:,}' for t in tk])
        ty = [t for t in tk if t <= top_k]
        ax.set_yticks(np.log10(ty)); ax.set_yticklabels([f'{t:,}' for t in ty])
        ax.set_xlim(lim, -0.1); ax.set_ylim(np.log10(top_k * 1.15), -0.1)
        r = spearmanr(sh.ex_rank, sh.de_rank)
        ax.set_title(f"{grp.split('|')[0]} hepatocytes\nSpearman ρ = {r.statistic:.2f}".replace('-', '−'),
                     color=VS.INK, fontweight='bold')
        ax.set_xlabel('DE rank (|Wald statistic|)'); ax.set_ylabel(f'ExPert rank (top {top_k})')
        ax.grid(False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        src.append(dict(group=grp, n_genes=n, shown=len(sh), rho=r.statistic, p=r.pvalue,
                        median_de_rank_of_shown=sh.de_rank.median(), shown_in_top_deg=int(de_m.sum())))
    h = [Line2D([], [], ls='', marker='o', ms=8, mfc=EXPERT, mec=VS.INK, label=f'ExPert (top-{mark})'),
         Line2D([], [], ls='', marker='o', ms=8, mfc=VS.INK, mec='none', label=f'DEG (top-{mark})'),
         Line2D([], [], ls='', marker='o', ms=6, mfc=VS.MUTED, mec='none', label=f'ExPert (top-{top_k})')]
    np.atleast_1d(axs)[0].legend(handles=h, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.3)
    save(VS, fig, out, name)
    pd.DataFrame(src).to_csv(SRC / f'{name}_source.csv', index=False)
    print(pd.DataFrame(src).round(3).to_string(index=False))


def volcano(VS, d, out, a, groups, name):
    """Own-transcript DE volcano per group (donor-level pseudobulk DESeq2: shrunk log2FC, BH-adjusted p) on plain
    linear axes: top-n DEGs (ink) and top-n ExPert predictions (ExPert orange; faded unless the gene passes BOTH
    cutoffs); dark dashed lines at |log2FC| = a.deg_lfc and adjusted p = 0.05."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    deg = pd.read_csv(a.deg)
    gd = pd.read_csv(d / 'fig5d_genes.csv.gz')
    n_ = a.volcano_n
    fig, axs = plt.subplots(len(groups), 1, figsize=(7.4, 6.2 * len(groups)), gridspec_kw={'hspace': 0.3})  # stacked
    pro_b = pd.read_csv(SRC / 'fig5b_umap_source.csv')     # 5b's prototypes; labelled = highlight set
    pro_b = pro_b[pro_b.highlight.notna() & (pro_b.highlight.astype(str) != '')]
    src, hl_rows = [], []
    xl = deg[deg.group.isin(groups)].logfc.abs().max() * 1.06       # one x range for every panel
    for ax, grp in zip(np.atleast_1d(axs), groups):
        x = deg[deg.group == grp].copy()
        x['nlp'] = -np.log10(x.pval_adj.clip(lower=1e-300))
        x['deg'] = (x.pval_adj < 0.05) & (x.logfc.abs() >= a.deg_lfc)
        top_deg = set(x[x.deg].assign(a_=x.score.abs()).nlargest(n_, 'a_').gene)
        top_mod = set(gd[(gd.group == grp)].nsmallest(n_, 'model_rank').gene)
        hl_rows += [dict(group=grp, gene=g_, set='DEG') for g_ in sorted(top_deg)] + \
                   [dict(group=grp, gene=g_, set='ExPert') for g_ in sorted(top_mod)]   # reused by 5g
        dd = x[x.gene.isin(top_deg)]
        mm = x[x.gene.isin(top_mod)]
        ax.scatter(x.logfc, x.nlp, s=8, color=VS.EXCLUDED, lw=0, zorder=1)
        ax.scatter(dd.logfc, dd.nlp, s=34, color=VS.INK, edgecolor=VS.SURFACE, lw=0.7, zorder=3)
        for ok, al, z in ((False, 1.0, 5), (True, 0.35, 4)):            # ExPert hits that ARE DEGs are faded
            m_ = mm[mm.deg == ok]
            ax.scatter(m_.logfc, m_.nlp, s=46, color=EXPERT, edgecolor=VS.INK, lw=0.7, alpha=al, zorder=z)
        for v in (-a.deg_lfc, a.deg_lfc):
            ax.axvline(v, color=VS.INK2, ls='--', lw=1.2, zorder=2)
        ax.axhline(-np.log10(0.05), color=VS.INK2, ls='--', lw=1.2, zorder=2)
        ax.set_xlim(-xl, xl)
        yl = max(x.nlp.max(), 2.0) * 1.08
        ax.set_ylim(-0.02 * yl, yl)
        # Only the genes 5b labels get text (5b labels no DEGs). ExPert labels = the SAME genes 5b labels for this
        # condition (prototypes in mRNA export / splicing or a literature pathway), wherever they fall here: genes left of 0 in
        # a stacked column at the left edge, the rest at the right edge, each ordered by height (lines do not cross);
        # ExPert-orange text, faded where the gene is a DEG (as its dot)
        cd0 = grp.split('|')[0]
        bset = set(pro_b[pro_b.arm == cd0].gene)
        col_t = []
        lb = x[x.gene.isin(bset)]
        for sub, xt, ha, rp in ((lb[lb.logfc < 0], -xl * 0.97, 'left', (1, 0.5)),
                                (lb[lb.logfc >= 0], xl * 0.97, 'right', (0, 0.5))):
            sub = sub.sort_values('nlp', ascending=False)
            ys_ = np.linspace(0.55, 0.55 - 0.038 * (len(sub) - 1), len(sub)) if len(sub) > 1 else [0.55]
            for yt, r_ in zip(ys_, sub.itertuples()):
                al_ = 0.35 if r_.deg else 1.0                                  # same opacity as the dot
                col_t.append(ax.annotate(r_.gene, (r_.logfc, r_.nlp), xytext=(xt, yl * yt), fontstyle='italic',
                            fontsize=VS.FONT_MIN - 4, color=EXPERT, alpha=al_, va='center', ha=ha, zorder=6,
                            arrowprops=dict(arrowstyle='-', color=EXPERT, alpha=al_ * 0.7, lw=0.5, shrinkA=1,
                                            shrinkB=3, relpos=rp)))
        cd_ = grp.split('|')[0]
        ax.set_title(f'{cd_} hepatocytes', color=VS.INK, fontweight='bold')
        ax.set_xlabel('own-transcript log2 fold change (vs healthy)')
        ax.set_ylabel('−log10 adjusted p')
        src.append(dict(group=grp, top_n=n_, n_deg=int(x.deg.sum()), top_deg=len(top_deg), top_model=len(top_mod),
                        model_in_table=len(mm), model_that_are_DEGs=int(mm.deg.sum()),
                        model_pass_p=int((mm.pval_adj < 0.05).sum()),
                        model_pass_lfc=int((mm.logfc.abs() >= a.deg_lfc).sum()), overlap=len(top_deg & top_mod)))
    h = [Line2D([], [], ls='', marker='o', ms=8, mfc=VS.INK, mec='none', label=f'DEG (top-{n_})'),
         Line2D([], [], ls='', marker='o', ms=9, mfc=EXPERT, mec=VS.INK, label=f'ExPert (top-{n_})')]
    np.atleast_1d(axs)[0].legend(handles=h, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.3)
    save(VS, fig, out, name)
    pd.DataFrame(hl_rows).to_csv(SRC / f'{name}_highlight_genes.csv', index=False)
    pd.DataFrame(src).to_csv(SRC / f'{name}_source.csv', index=False)
    print(pd.DataFrame(src).to_string(index=False))


# 6i: the grouped mRNA export set (6c's top row) split into its machinery families
EXPORT_FAMILIES = [('NUP complex', 'NUP*, TPR, RAE1, …'), ('TREX / TREX-2', 'THOC*, SARNP, …'),
                   ('Ran cycle', 'RAN, RANGAP1, RCC1, …'), ('EJC / cap-binding', 'NCBP*, CASC3, …'),
                   ('SR proteins', 'SRSF*'), ('exportins', 'XPO*, NXF*')]


def export_family(g):
    if g.startswith(('NUP', 'POM121')) or g in ('AAAS', 'NDC1', 'RAE1', 'SEH1L', 'TPR', 'RANBP2', 'GLE1', 'DDX19A',
                                               'DDX19B'):
        return 'NUP complex'
    if g.startswith('THOC') or g in ('DDX39A', 'DDX39B', 'CHTOP', 'POLDIP3', 'SARNP', 'FYTTD1', 'MCM3AP', 'ENY2'):
        return 'TREX / TREX-2'
    if g.startswith('SRSF') or g in ('SRRM1', 'RNPS1'):
        return 'SR proteins'
    if g in ('CASC3', 'EIF4A3', 'MAGOH', 'MAGOHB', 'RBM8A', 'NCBP1', 'NCBP2', 'SRRT', 'PHAX', 'UPF3B', 'EIF4E'):
        return 'EJC / cap-binding'
    if g.startswith(('NXF', 'NXT', 'XPO')) or g == 'CSE1L':
        return 'exportins'
    if g in ('RAN', 'RANBP1', 'RANGAP1', 'RCC1', 'NUTF2'):
        return 'Ran cycle'
    return 'other'


def family_bars(VS, d, out, cc, a, gs, name='fig5l_export_families', pair=False):
    """6i: the grouped mRNA export set split into machinery families. Per family x condition, a box over the members'
    hepatocyte loss z (prediction score standardised over the bank; > 0 = the state resembles losing the gene; ALD z =
    mean of AC and AH z) with every member as a dot; members ExPert names (ALD top-a.top loss union) are drawn larger
    with a dark ring. Stars vs healthy only: paired Wilcoxon over ALL family members (condition z vs healthy z; testing
    only the named members would be circular -- they were selected for being high in AC / AH), two-sided, BH over
    families x 3 conditions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import wilcoxon
    e = np.load(d / 'fig5_profiles.npz', allow_pickle=True)
    L = np.array([str(x) for x in e['labels']])
    S = {str(k): e['cond'][i] for i, k in enumerate(e['cond_keys'])}
    z = {c: (S[f'{c}|Hepatocyte'] - S[f'{c}|Hepatocyte'].mean()) / S[f'{c}|Hepatocyte'].std() for c in ('Normal', 'AC', 'AH')}
    z['ALD'] = (z['AC'] + z['AH']) / 2
    named = set().union(*[set(L[np.argsort(-S[f'{c}|Hepatocyte'], kind='stable')[:a.top]]) for c in ('AC', 'AH')])
    pos = {g: i for i, g in enumerate(L)}
    mem = [g for g in gs['mRNA export (grouped)'] if g in pos]
    M = pd.DataFrame([dict(gene=g, family=export_family(g), named=g in named,
                           **{c: float(z[c][pos[g]]) for c in ('Normal', 'ALD', 'AC', 'AH')}) for g in mem])
    M.to_csv(SRC / f'{name}_members.csv', index=False)
    conds = [('Normal', 'healthy'), ('ALD', 'ALD')] + ([] if pair else [('AC', 'AC'), ('AH', 'AH')])
    tests = []
    for f, _ in EXPORT_FAMILIES:
        sub = M[M.family == f]
        for cd in [c for c, _ in conds[1:]]:
            tests.append(dict(family=f, cond=cd, k=len(sub), named=int(sub.named.sum()),
                              mean_z=sub[cd].mean(), mean_healthy=sub.Normal.mean(),
                              p=wilcoxon(sub[cd], sub.Normal).pvalue))
    T = pd.DataFrame(tests)
    pv = T.p.to_numpy(); o_ = np.argsort(pv)
    q = pv[o_] * len(pv) / np.arange(1, len(pv) + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    qq = np.empty_like(q); qq[o_] = np.minimum(q, 1); T['p_bh'] = qq
    T.to_csv(SRC / f'{name}_tests.csv', index=False)
    star = lambda p: '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''
    nf = len(EXPORT_FAMILIES)
    fig, ax = plt.subplots(figsize=(4.6, (1.15 if pair else 1.75) * nf + 1.0))
    wb, rng = 0.21, np.random.default_rng(0)
    colour = {'healthy': cc['healthy'], 'ALD': ALD_C, 'AC': cc['AC'], 'AH': cc['AH']}
    xr = M[['Normal', 'ALD', 'AC', 'AH']].to_numpy()
    xhi = np.nanmax(xr)
    # healthy / ALD tall, AC / AH slimmer and stacked below them (the focus is healthy vs ALD)
    hgt = {'healthy': 0.24, 'ALD': 0.24, 'AC': 0.13, 'AH': 0.13}
    off = {'healthy': -0.33, 'ALD': -0.07, 'AC': 0.15, 'AH': 0.31}
    if pair:
        hgt = {'healthy': 0.36, 'ALD': 0.36}
        off = {'healthy': -0.2, 'ALD': 0.2}
    for i, (f, _) in enumerate(EXPORT_FAMILIES):
        sub = M[M.family == f]
        for j, (cd, ck) in enumerate(conds):
            y = i + off[ck]
            v = sub[cd].to_numpy()
            bp = ax.boxplot([v], positions=[y], vert=False, widths=hgt[ck] * 0.9, showfliers=False, patch_artist=True,
                            medianprops=dict(color=VS.INK, lw=1.4), whiskerprops=dict(color=VS.INK2, lw=1.0),
                            capprops=dict(color=VS.INK2, lw=1.0), zorder=2)
            bx = bp['boxes'][0]
            if ck == 'ALD':
                bx.set_facecolor('none'); bx.set_edgecolor(ALD_C); bx.set_linewidth(1.6)
            else:
                bx.set_facecolor(colour[ck]); bx.set_alpha(0.3); bx.set_edgecolor(VS.INK2)
            small = ck in ('AC', 'AH')
            jit = rng.uniform(-hgt[ck] * 0.25, hgt[ck] * 0.25, len(v))
            nm = sub.named.to_numpy()
            ax.scatter(v[~nm], y + jit[~nm], s=6 if small else 12, color=colour[ck], lw=0, zorder=3)
            ax.scatter(v[nm], y + jit[nm], s=16 if small else 34, color=colour[ck], edgecolor=VS.INK,
                       lw=0.7 if small else 0.9, zorder=4)
            if cd != 'Normal':
                r_ = T[(T.family == f) & (T.cond == cd)].iloc[0]
                st_ = star(r_.p if pair else r_.p_bh)                # main: uncorrected, per family
                if st_:
                    ax.text(xhi + 0.25, y, st_, ha='left', va='center', color=VS.INK, fontsize=VS.FONT_MIN)
    ax.axvline(0, color=VS.INK2, lw=1.0, zorder=1)
    ax.set_yticks(range(nf))
    ax.set_yticklabels([f for f, _ in EXPORT_FAMILIES], va='bottom')
    for tl in ax.get_yticklabels():
        tl.set_color(VS.INK)
    for i, (_, g) in enumerate(EXPORT_FAMILIES):                     # example genes: smaller, lighter, below
        ax.text(-0.025, i + 0.04, g, transform=ax.get_yaxis_transform(), ha='right', va='top', color=VS.MUTED,
                fontsize=VS.FONT_MIN * 0.8)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(nf - 0.45, -0.55)
    ax.set_xlim(np.nanmin(xr) - 0.3, xhi + 1.1)
    ax.set_xlabel('predicted loss (z)')
    ax.grid(axis='y', visible=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    h = [Patch(facecolor=colour['healthy'], alpha=0.3, edgecolor=VS.INK2, label='healthy'),
         Patch(facecolor='none', edgecolor=ALD_C, lw=1.6, label='ALD'),
         Patch(facecolor=colour['AC'], alpha=0.3, edgecolor=VS.INK2, label='AC'),
         Patch(facecolor=colour['AH'], alpha=0.3, edgecolor=VS.INK2, label='AH'),
         Line2D([], [], ls='', marker='o', ms=6, mfc=VS.MUTED, mec=VS.INK, mew=0.9, label='named by ExPert')]
    if pair:
        h = [h[0], h[1], h[4]]
        marker_title(VS, ax, 'mRNA export predictions', 'D', EXPERT, y=1.02)
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.3, -0.06 - 0.3 / nf), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=0.8)
    save(VS, fig, out, name)
    print(T.assign(p_bh=T.p_bh.round(4), mean_z=T.mean_z.round(2)).to_string(index=False))


def plot_setup(a, VS=None):
    """Shared state of the plot stage (style, colours, the ORA tables, gene sets); sets the module SRC. The notebook
    (paper/figures/fig5/fig5.ipynb) calls this once and then the per-panel functions below; stage_plot draws all."""
    from types import SimpleNamespace
    import matplotlib
    if VS is None:
        sys.path.insert(0, str(Path(a.vizstyle_root)))
        from src import vizstyle as VS
    VS.apply()
    matplotlib.rcParams['pdf.fonttype'] = 42
    global SRC
    d, out = Path(a.out_dir), Path(a.fig_dir)
    SRC = Path(a.src_dir); SRC.mkdir(parents=True, exist_ok=True)
    # AH violet, never orange (orange = ExPert across the paper); AC blue vs AH violet: CVD dE 13.0, normal 16.3
    cc = {'Normal': '#a9a8a2', 'healthy': '#8a8983', 'AC': AC_C, 'AH': AH_C, 'ALD': ALD_C}
    colour_of = lambda c: (cc[c[:2]] if c[:2] in ('AC', 'AH') else cc['healthy'] if c.startswith(('Nor', 'healthy'))
                           else cc.get(c, VS.INK))
    T = pd.read_csv(d / 'fig5b_ora.csv.gz')
    plabels = [str(x) for x in np.load(d / 'fig5_profiles.npz', allow_pickle=True)['labels']]
    gs = gene_sets(a, plabels)
    mem_m = {k: set(v) for k, v in gs.items() if k.startswith('Reactome: ') or k in dict(HIGHLIGHT).values()}
    TD = pd.read_csv(d / 'fig5b_ora_deg.csv.gz')
    mem_d = {k: set(str(v).split(';')) for k, v in TD.drop_duplicates('set').set_index('set').members.items()}
    return SimpleNamespace(VS=VS, d=d, out=out, cc=cc, colour_of=colour_of, T=T, TD=TD, plabels=plabels, gs=gs,
                           mem_m=mem_m, mem_d=mem_d)


def collapsed_rows(c, a):
    """5E / 5F rows deduplicated by HIT overlap (hits = set genes in the ALD top-200 union of the best tail):
    ((5E rows, n more, folded), (5F rows, n more, folded))"""
    d, T, TD = c.d, c.T, c.TD
    lab_ = np.asarray(c.plabels)
    pe = np.load(d / 'fig5_profiles.npz', allow_pickle=True)
    Sc_ = {str(k): pe['cond'][i] for i, k in enumerate(pe['cond_keys'])}
    U_m = {t: set().union(*[set(lab_[np.argsort(-sg_ * Sc_[k], kind='stable')[:a.top]]) for k in
                            ('AC|Hepatocyte', 'AH|Hepatocyte')]) for t, sg_ in (('loss', 1), ('gain', -1))}
    deg_ = pd.read_csv(a.deg)
    H_ = deg_[deg_.group.isin(['AC|Hepatocyte', 'AH|Hepatocyte'])]
    U_d = {t: set().union(*[set(H_[H_.group == g].sort_values('score', ascending=asc).gene.head(a.top))
                            for g in ('AC|Hepatocyte', 'AH|Hepatocyte')]) for t, asc in (('loss', True), ('gain', False))}
    bt_ = lambda X: X[X.column == 'ALD'].loc[lambda z: z.groupby('set').p.idxmin()].set_index('set')['tail']
    def hit_of(mem, U, X):
        b_ = bt_(X)                                          # best tail per set, computed ONCE (was per set: ~400 s)
        return {k: v & U[b_.get(k, 'loss')] for k, v in mem.items()}
    fo_m, fo_d = {}, {}
    r_m, _, mo_m = top_rows(T, 'ALD', c.mem_m, a.n_top, n_more_p=0.05, hits=hit_of(c.mem_m, U_m, T), folded=fo_m,
                            order='fe')
    r_d, _, mo_d = top_rows(TD, 'ALD', c.mem_d, a.n_top, n_more_p=0.05, highlight=False,
                            hits=hit_of(c.mem_d, U_d, TD), folded=fo_d, order='fe')
    return (r_m, mo_m, fo_m), (r_d, mo_d, fo_d)


def panel_pathways(c, rows, n_more):
    """5E: the collapsed ExPert rows, healthy and ALD hepatocyte columns"""
    column_dots(c.VS, [('healthy', c.T, 'healthy'), ('ALD', c.T, 'ALD')], rows, n_more, c.out, 'fig5e_pathways',
                title=('ExPert', 'D', EXPERT))


def panel_pathways_deg(c, rows, n_more):
    """5F: the collapsed DE rows, ALD hepatocytes"""
    column_dots(c.VS, [('ALD', c.TD, 'ALD')], rows, n_more, c.out, 'fig5f_pathways_deg', title=('DEG', 'o', c.VS.INK2))


def umap_keys(c):
    """5B label sets: the highlighted families + the literature rows enriched in ALD"""
    rows, _, _ = best_tail_rows(c.T, 'ALD')
    return [k for _, k in HIGHLIGHT] + [k for _, k, blk in rows if blk == 'literature']


def panel_umap(c, a):
    """5B: healthy vs ALD nuclei + top prototypes (writes SRC/fig5b_umap_source.csv, which 5D reads)"""
    umap_plot(c.VS, c.d, c.out, 'scoring', 'fig5b_umap', c.gs, umap_keys(c), c.cc, a, pooled=True)


def stage_plot(a):
    import matplotlib
    matplotlib.use('Agg')
    c = plot_setup(a)
    VS, d, out, cc, gs = c.VS, c.d, c.out, c.cc, c.gs
    # E / F: rows deduplicated by HIT overlap (hits = set genes in the ALD top-200 union of the best tail)
    (r_m, mo_m, fo_m), (r_d, mo_d, fo_d) = collapsed_rows(c, a)
    panel_pathways(c, r_m, mo_m)
    panel_pathways_deg(c, r_d, mo_d)
    pd.DataFrame([dict(panel='6b', set=k, folded_into=v) for k, v in fo_m.items()] +
                 [dict(panel='6d', set=k, folded_into=v) for k, v in fo_d.items()]).to_csv(
        SRC / 'fig5bd_collapsed_folds.csv', index=False)
    orth_stacked(VS, d, out, a, gs, cc)
    family_bars(VS, d, out, cc, a, gs, name='fig5l_export_families', pair=True)
    panel_umap(c, a)                                        # B: healthy vs ALD
    volcano(VS, d, out, a, a.d_groups, 'fig5d_volcano')     # after B: labels exactly B's labelled genes


def umap_plot(VS, d, out, var, name, gs, keys, cc, a, pooled=False):
    """Nuclei by condition + top prototypes; only members of mRNA export or mRNA splicing (incl. SR proteins) are
    labelled. pooled: healthy vs ALD (AC + AH) nuclei, every top
    prototype in the ALD colour."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from adjustText import adjust_text
    cells = pd.read_csv(d / f'fig5a_cells_{var}.csv.gz')
    pro = pd.read_csv(d / f'fig5a_prototypes_{var}.csv')
    fig, ax = plt.subplots(figsize=(7, 7))
    col_of = (lambda c: c.map(lambda v: cc['Normal'] if v == 'Normal' else cc['ALD'])) if pooled else \
        (lambda c: c.map(cc))
    for sub in (cells[cells.condition == 'Normal'], cells[cells.condition != 'Normal'].sample(frac=1, random_state=0)):
        ax.scatter(sub.umap1, sub.umap2, s=2.5, lw=0, c=col_of(sub.condition), alpha=0.55)
    for arm in ('AC', 'AH'):
        p_ = pro[pro.arm == arm]
        if pooled:                                   # ExPert-coloured, filled (they are ExPert predictions)
            ax.scatter(p_.umap1, p_.umap2, s=75, marker='D', color=EXPERT, edgecolor=VS.INK, lw=1.2, zorder=4)
        else:
            ax.scatter(p_.umap1, p_.umap2, s=70, marker='D', color=cc[arm], edgecolor=VS.INK, lw=1.2, zorder=4)
    # label prototypes in mRNA export / splicing (bold) or a literature pathway (normal); every label has a leader line
    hl_sets = [k for _, k in HIGHLIGHT]
    member = {g: [k for k in keys if g in set(gs.get(k, []))] for g in pro.gene}
    pro['highlight'] = [';'.join(member[g]) for g in pro.gene]
    bold = {g for g in pro.gene if any(g in set(gs.get(k, [])) for k in hl_sets)}
    pl = pro[pro.highlight != '']
    ts = [ax.text(r_.umap1, r_.umap2, r_.gene, fontstyle='italic', fontsize=a.label_size, zorder=6, color=VS.INK,
                  fontweight='bold' if r_.gene in bold else 'normal') for r_ in pl.itertuples()]
    adjust_text(ts, x=pro.umap1.to_numpy(), y=pro.umap2.to_numpy(), ax=ax, expand=(1.6, 1.9), min_arrow_len=0,
                force_text=(0.5, 0.8), force_static=(0.6, 0.9),
                arrowprops=dict(arrowstyle='-', color=VS.INK2, lw=0.7, shrinkA=0, shrinkB=3))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    if pooled:
        h = [Line2D([], [], ls='', marker='o', ms=9, color=cc[c], label=f'{lab} nuclei')
             for c, lab in (('Normal', 'healthy'), ('ALD', 'ALD'))]
        h += [Line2D([], [], ls='', marker='D', ms=9, mfc=EXPERT, mec=VS.INK, mew=1.2, label='top ALD prototype')]
    else:
        h = [Line2D([], [], ls='', marker='o', ms=9, color=cc[c], label=f'{c} nuclei') for c in ('Normal', 'AC', 'AH')]
        h += [Line2D([], [], ls='', marker='D', ms=9, color=cc[c], mec=VS.INK, mew=1.2, label=f'top {c} prototype')
              for c in ('AC', 'AH')]
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.3)
    save(VS, fig, out, name)
    pro.to_csv(SRC / f'{name}_source.csv', index=False)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['data', 'plot'])
    ap.add_argument('--out-dir', required=True, help='data dir (paper/figures/fig5/data/final)')
    ap.add_argument('--fig-dir', help='figure dir (fig5)')
    ap.add_argument('--src-dir',
                    help='plot: source tables behind the panels (paper/figures/fig5/data)')
    ap.add_argument('--predict-dir', help='fig5_predict.py encode output (predict_groups.npz, obs_groups.csv)')
    ap.add_argument('--cells-dir', help='fig5_readout_chain.py encode output (enc__ct_normal.npz) for the UMAP')
    ap.add_argument('--bank-npz')
    ap.add_argument('--deg', help='readout_deg/deg.csv.gz')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--modules')
    ap.add_argument('--vizstyle-root')
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--top-n', type=int, default=50, help='6d: model top-n for the DEG overlap')
    ap.add_argument('--n-atoms', type=int, default=50, help='6a: distinct top prototypes over AC + AH')
    ap.add_argument('--umap-centring', nargs='+', default=list(UMAP_CENTRING), help='6a latent centrings')
    ap.add_argument('--umap-only', action='store_true', help='data: only (re)compute the 6a UMAPs')
    ap.add_argument('--label-size', type=float, default=9.5, help='6a: prototype label font size')
    ap.add_argument('--d-groups', nargs='+', default=['AC|Hepatocyte', 'AH|Hepatocyte'], help='6d scatters')
    ap.add_argument('--n-top', type=int, default=15, help='6b / 6d: unbiased top-n pathways')
    ap.add_argument('--deg-lfc', type=float, default=2.0, help='DEG = padj < 0.05 and |log2FC| >= this (6c volcano, 6S overlap)')
    ap.add_argument('--volcano-n', type=int, default=100, help='6c volcano: top-n DEGs and top-n ExPert predictions')
    ap.add_argument('--skip-umap', action='store_true', help='data: tables only')
    return ap


if __name__ == '__main__':
    a = parser().parse_args()
    {'data': stage_data, 'plot': stage_plot}[a.stage](a)
