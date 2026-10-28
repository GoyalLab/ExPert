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

"""Fig 5g / 5k: the bulk RNA-seq livers of Chembazhi 2025 (4 healthy, 6 AC, 6 AH; donor overlap with the snRNA-seq pools unknown; DESeq2 + rMATS tables).

    fig5g_bulk_volcano   AC and AH vs healthy (stacked): log2FC vs -log10 padj over all genes passing the count filter;
                         the ExPert top-n hepatocyte predictions (loss tail, same condition) in ExPert orange -- faded
                         where the gene IS a bulk DEG (padj < .05 and |log2FC| >= --deg-lfc), as in 5d
    fig5k_splicing       genes with rMATS events (2 x 2: columns AC / AH; rows DE rank and ExPert rank vs splicing
                         rank); splicing rank = the gene's most significant event (min FDR, ties: max |dPSI|), DE rank
                         = |Wald| in bulk, ExPert rank = hepatocyte loss score; all ranks within the shown genes;
                         mRNA export / splicing genes coloured as 5c. Spearman rho -> <--src-dir>/fig5k_splicing_source.csv

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_bulk.py --bulk-dir .../BulkRNASeqAnalysis/tables --src-dir ... ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

CONDS = ['AC', 'AH']


def expert_scores(profiles, cond):
    e = np.load(profiles, allow_pickle=True)
    L = [str(x) for x in e['labels']]
    S = {str(k): e['cond'][i] for i, k in enumerate(e['cond_keys'])}
    return pd.Series(S[f'{cond}|Hepatocyte'], index=L)


def bulk_de(bulk_dir, cond):
    d = pd.read_csv(Path(bulk_dir) / f'DESeq2_all_genes_Healthy_vs_{cond}.csv', low_memory=False)
    d = d[d.passed_prefilter.astype(str).str.upper().eq('TRUE') & d.padj.notna() & d.SYMBOL.notna()]
    d = d.assign(gene=d.SYMBOL.str.split('; ').str[0])
    return d.sort_values('pvalue').drop_duplicates('gene')


def volcano(VS, F, a, out, gs):
    """5g: AC / AH bulk volcanos (stacked). Every gene coloured by ExPert's hepatocyte predicted-loss z for that
    condition (snRNA-seq score standardised over the bank; z <= 0 light grey -> ExPert orange at z >= a.z_max),
    strongest on top; genes ExPert cannot score in grey. Every ExPert top-a.top_n gene in the mRNA export / splicing
    sets is named in ExPert-orange text (a.n_label > 0 keeps only the top a.n_label). No legend: one colour bar."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    ann = set(gs['mRNA export (grouped)']) | set(gs['mRNA splicing'])
    cmap = LinearSegmentedColormap.from_list('expert_z', ['#e6e5e1', F.EXPERT])
    norm = Normalize(vmin=0.0, vmax=a.z_max)
    fig, axs = plt.subplots(len(CONDS), 1, figsize=(7.4, 6.2 * len(CONDS)), gridspec_kw={'hspace': 0.3})
    src = []
    des = {c: bulk_de(a.bulk_dir, c) for c in CONDS}
    xl = max(d.log2FoldChange.abs().quantile(0.999) for d in des.values()) * 1.1
    for i, (ax, c) in enumerate(zip(axs, CONDS)):
        d = des[c].assign(nlp=lambda z: -np.log10(z.padj.clip(lower=1e-300)))
        d['x'] = d.log2FoldChange.clip(-xl, xl)
        d['deg'] = (d.padj < 0.05) & (d.log2FoldChange.abs() >= a.deg_lfc)
        sc0 = expert_scores(a.profiles, c)
        d['z'] = d.gene.map((sc0 - sc0.mean()) / sc0.std())
        sc = sc0[sc0.index.isin(set(d.gene))].sort_values(ascending=False)
        rank = pd.Series(np.arange(1, len(sc) + 1), index=sc.index)
        top = d[d.gene.isin(set(sc.index[:a.top_n]))].assign(rank=lambda z: z.gene.map(rank))
        hi_ = top[top.gene.isin(ann)]
        na = d[d.z.isna()]
        sd = d[d.z.notna()].sort_values('z')                         # strongest predictions drawn last
        ax.scatter(na.x, na.nlp, s=5, color=VS.EXCLUDED, lw=0, zorder=1)
        ax.scatter(sd.x, sd.nlp, s=np.where(sd.z >= 2, 22, 7), c=sd.z.clip(lower=0), cmap=cmap, norm=norm, lw=0,
                   zorder=2)
        for v in (-a.deg_lfc, a.deg_lfc):
            ax.axvline(v, color=VS.INK2, ls='--', lw=1.2, zorder=3)
        ax.axhline(-np.log10(0.05), color=VS.INK2, ls='--', lw=1.2, zorder=3)
        ymax = d.nlp.max() * 1.06
        # every ExPert top-n export / splicing gene named (ExPert-orange text): genes left of 0 in a stacked column at
        # the left edge, the rest at the right edge; each column ordered by height so leader lines do not cross
        lab = hi_ if a.n_label <= 0 else hi_.nsmallest(a.n_label, 'rank')
        for sub, xt, ha, rp in ((lab[lab.x < 0], -xl * 0.97, 'left', (1, 0.5)),
                                (lab[lab.x >= 0], xl * 0.97, 'right', (0, 0.5))):
            sub = sub.sort_values('nlp', ascending=False)
            ys = np.linspace(0.92, 0.92 - 0.045 * (len(sub) - 1), len(sub)) if len(sub) > 1 else [0.92]
            for yt, r in zip(ys, sub.itertuples()):
                ax.annotate(r.gene, (r.x, r.nlp), xytext=(xt, ymax * yt), fontstyle='italic',
                            fontsize=VS.FONT_MIN - 4, color=F.EXPERT, ha=ha, va='center', zorder=6,
                            arrowprops=dict(arrowstyle='-', color=VS.INK2, lw=0.5, alpha=0.7, shrinkA=1, shrinkB=3,
                                            relpos=rp))
        ax.set_xlim(-xl, xl)
        ax.set_ylim(-0.02 * ymax, ymax)
        ax.set_title(f'{c} vs healthy (bulk RNA-seq)', color=VS.INK, fontweight='bold')
        ax.set_xlabel('log2 fold change'); ax.set_ylabel('−log10 adjusted p')
        ax.grid(False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if i == 0:
            cax = ax.inset_axes([1.05, 0.45, 0.035, 0.5])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='max')
            cb.outline.set_visible(False)
            cb.set_label('ExPert predicted loss (z)', color=VS.INK2)
        src.append(dict(cond=c, genes_tested=len(d), scored_by_expert=int(d.z.notna().sum()), n_deg=int(d.deg.sum()),
                        expert_top=a.top_n, expert_that_are_bulk_DEGs=int(top.deg.sum()),
                        module_genes=int(d.gene.isin(ann).sum()), module_DEGs=int(d[d.gene.isin(ann)].deg.sum()),
                        export_splicing=len(hi_), export_splicing_that_are_DEGs=int(hi_.deg.sum()),
                        expert_median_abs_lfc=top.log2FoldChange.abs().median(),
                        all_median_abs_lfc=d.log2FoldChange.abs().median(), named=';'.join(lab.gene)))
    F.save(VS, fig, out, 'fig5g_bulk_volcano')
    S = pd.DataFrame(src); S.to_csv(Path(a.src_dir) / 'fig5g_bulk_volcano_source.csv', index=False)
    print(S.drop(columns=['named']).round(3).to_string(index=False))


VALIDATED = ['AKAP8L', 'SRSF3', 'POLDIP3', 'XPO1', 'NUP42', 'NUP54', 'NUP214', 'UPF3A']   # RT-PCR screen (5m)


def sig_splice_genes(a, c):
    """Genome-wide significant splicing genes -> |dPSI| of their representative significant event (the collaborators'
    rMATS gene table: in_significant_rmats = FDR < 0.01 and |dPSI| >= 0.1). NOT the Fig6_rMATS_event_statistics
    tables -- those only hold events of the genes in our earlier ExPert export (fig6_top200_per_group.csv)."""
    g = pd.read_csv(Path(a.bulk_dir) / f'Fig6_rMATS_gene_overlap_{c}_vs_Healthy.csv', low_memory=False)
    g = g[g.in_significant_rmats.astype(str).str.upper() == 'TRUE']
    return g.groupby('gene_label').representative_dpsi.apply(lambda v: v.abs().max())


def family_of(F, exp_set):
    """Gene -> family: the 5l machinery families for mRNA-export genes, else the HGNC-style root symbol (letters,
    >= 3; NUP153 -> NUP*). A loss of any member of a complex / family looks alike to the model."""
    import re

    def f(g):
        if g in exp_set:
            fm = F.export_family(g)
            if fm != 'other':
                return fm
        m = re.match(r'^([A-Z]+)', g)
        return (m.group(1) if m and len(m.group(1)) >= 3 else g) + '*'
    return f


def splice_volcano(VS, F, a, out, gs):
    """5k: bulk volcanos (AC / AH stacked, as 5d / 5g); significantly spliced genes coloured by |dPSI| (colour bar),
    RT-PCR-screened genes (5m) labelled. Below, ExPert's top-a.splice_k hepatocyte predictions in rank order: filled
    if ANY member of the gene's family is significantly spliced AND the gene itself is not a bulk DEG; observed count vs a random-gene-set null (same k,
    same family mapping, universe = bank genes passing the bulk count filter), mRNA-export families in bold."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    exp_set = set(gs['mRNA export (grouped)'])
    fam = family_of(F, exp_set)
    cmap = LinearSegmentedColormap.from_list('dpsi', VS.SEQ[1:])
    norm = Normalize(vmin=0.1, vmax=0.6)
    fig = plt.figure(figsize=(7.4, 6.2 * len(CONDS) + 5.0))
    gsp = fig.add_gridspec(len(CONDS) + 1, 1, height_ratios=[6.2] * len(CONDS) + [3.9], hspace=0.36)
    sc_all = {c: expert_scores(a.profiles, c) for c in CONDS}
    des = {c: bulk_de(a.bulk_dir, c) for c in CONDS}
    xl = max(d.log2FoldChange.abs().quantile(0.999) for d in des.values()) * 1.1
    rng = np.random.default_rng(0)
    rows, strips = [], []
    for i, c in enumerate(CONDS):
        ax = fig.add_subplot(gsp[i, 0])
        d = des[c].assign(nlp=lambda z: -np.log10(z.padj.clip(lower=1e-300)))
        d['x'] = d.log2FoldChange.clip(-xl, xl)
        sp = sig_splice_genes(a, c)
        d['dpsi'] = d.gene.map(sp)
        ss = d.dpsi.notna()
        ax.scatter(d.x[~ss], d.nlp[~ss], s=6, color=VS.EXCLUDED, lw=0, zorder=1)
        o_ = d[ss].sort_values('dpsi')
        ax.scatter(o_.x, o_.nlp, s=14, c=o_.dpsi.clip(upper=0.6), cmap=cmap, norm=norm, edgecolor='white', lw=0.3,
                   zorder=2)
        for v in (-a.deg_lfc, a.deg_lfc):
            ax.axvline(v, color=VS.INK2, ls='--', lw=1.2, zorder=0)
        ax.axhline(-np.log10(0.05), color=VS.INK2, ls='--', lw=1.2, zorder=0)
        lab = d[d.gene.isin(VALIDATED)].sort_values('x')
        ax.scatter(lab.x, lab.nlp, s=46, facecolor='none', edgecolor=VS.INK, lw=1.1, zorder=4)
        top = d.nlp.max() * 1.06
        ys = np.linspace(0.78, 0.30, len(lab)) if len(lab) > 1 else [0.6]
        for yt, r in zip(ys, lab.itertuples()):
            ax.annotate(r.gene, (r.x, r.nlp), xytext=(-xl * 0.88, top * yt), fontstyle='italic',
                        fontsize=VS.FONT_MIN - 2, color=VS.INK, ha='left', va='center', zorder=6,
                        arrowprops=dict(arrowstyle='-', color=VS.INK2, lw=0.7, shrinkA=1, shrinkB=3))
        ax.set_xlim(-xl, xl); ax.set_ylim(-0.02 * top, top)
        ax.set_title(f'{c} vs healthy (bulk RNA-seq)', color=VS.INK, fontweight='bold')
        ax.set_xlabel('log2 fold change'); ax.set_ylabel('−log10 adjusted p')
        ax.grid(False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if i == 0:
            cax = ax.inset_axes([1.06, 0.45, 0.035, 0.45])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='max')
            cb.outline.set_visible(False)
            cb.set_label('|ΔPSI| (significant splicing)', color=VS.INK2)
            ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=8, mfc='none', mec=VS.INK, mew=1.1,
                                      label='screened by RT-PCR')],
                      loc='upper left', bbox_to_anchor=(1.0, 0.33), handletextpad=0.3, frameon=False)
        # family-level overlap of ExPert's top-k with significant splicing, vs random gene sets
        U = np.array(sorted(set(d.gene) & set(sc_all[c].index)))
        fams = np.array([fam(u) for u in U])
        hitfam = set(fams[np.isin(U, sp.index)])
        deg_ = set(d[(d.padj < 0.05) & (d.log2FoldChange.abs() >= a.deg_lfc)].gene)
        hit = np.isin(fams, list(hitfam)) & ~np.isin(U, list(deg_))   # family spliced AND gene itself not a bulk DEG
        order = np.argsort(-sc_all[c].reindex(U).to_numpy())
        k = a.splice_k
        top_ = order[:k]
        null = np.array([hit[rng.choice(len(U), k, replace=False)].sum() for _ in range(a.n_perm)])
        obs = int(hit[top_].sum())
        rows.append(dict(cond=c, k=k, universe=len(U), family_hit_rate=hit.mean(), observed=obs,
                         null_mean=null.mean(), null_lo=np.percentile(null, 2.5), null_hi=np.percentile(null, 97.5),
                         p=(null >= obs).mean(), exact_gene_hits=int(np.isin(U[top_], sp.index).sum()),
                         export_family_genes=int(np.isin(U[top_], list(exp_set)).sum()),
                         export_family_hits=int((np.isin(U[top_], list(exp_set)) & hit[top_]).sum())))
        strips.append((c, U[top_], fams[top_], hit[top_]))
    R = pd.DataFrame(rows)
    ax = fig.add_subplot(gsp[-1, 0])
    for j, (c, genes, fm, ht) in enumerate(strips):
        y = 1.9 * j                                       # room for the rotated gene names above each row
        xs = np.arange(1, len(genes) + 1)
        ax.scatter(xs[ht], np.full(ht.sum(), y), s=110, color=VS.INK, zorder=3)
        ax.scatter(xs[~ht], np.full((~ht).sum(), y), s=110, facecolor='white', edgecolor=VS.INK, lw=1.1, zorder=3)
        for x_, g_ in zip(xs, genes):
            ax.text(x_, y - 0.28, g_, rotation=90, ha='center', va='bottom', fontstyle='italic',
                    fontsize=VS.FONT_MIN - 5, color=VS.INK, fontweight='bold' if g_ in exp_set else 'normal')
        r_ = R.iloc[j]
        ax.text(len(genes) + 1.2, y, f"{int(r_.observed)}/{int(r_.k)}", ha='left', va='center', color=VS.INK2,
                fontsize=VS.FONT_MIN - 3)                 # null (random gene sets) not significant -> source table only
    ax.set_yticks([1.9 * j for j in range(len(strips))]); ax.set_yticklabels([c for c, *_ in strips])
    ax.set_ylim(1.9 * (len(strips) - 1) + 0.4, -1.45)
    ax.set_xlim(0.3, a.splice_k + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20][:sum(t <= a.splice_k for t in (1, 5, 10, 15, 20))])
    ax.set_xlabel('ExPert rank (hepatocytes, predicted loss)')
    ax.grid(False)
    for sp_ in ('top', 'right', 'left'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=9, color=VS.INK, label='family spliced, gene not a DEG'),
                       Line2D([], [], ls='', marker='o', ms=9, mfc='white', mec=VS.INK, mew=1.1, label='no')],
              loc='upper left', bbox_to_anchor=(1.12, 1.0), handletextpad=0.3, frameon=False)
    F.save(VS, fig, out, 'fig5k_splicing')
    R.to_csv(Path(a.src_dir) / 'fig5k_splicing_source.csv', index=False)
    print(R.round(3).to_string(index=False))


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    import matplotlib
    matplotlib.use('Agg')
    VS.apply()
    VS.SURFACE = 'white'
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                                'savefig.facecolor': 'white'})
    import fig5_final as F
    out = Path(a.fig_dir)
    Path(a.src_dir).mkdir(parents=True, exist_ok=True)
    L = [str(x) for x in np.load(a.profiles, allow_pickle=True)['labels']]
    gs = F.gene_sets(argparse.Namespace(modules=a.modules, reactome_gmt=a.reactome_gmt), L)
    volcano(VS, F, a, out, gs)
    splice_volcano(VS, F, a, out, gs)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bulk-dir', required=True, help='BulkRNASeqAnalysis/tables')
    ap.add_argument('--profiles', required=True, help='data/final/fig5_profiles.npz (ExPert condition profiles)')
    ap.add_argument('--fig-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--modules', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--top-n', type=int, default=200, help='5g: ExPert top-n hepatocyte predictions shown')
    ap.add_argument('--z-max', type=float, default=3.0, help='5g: ExPert z at full ExPert orange')
    ap.add_argument('--n-label', type=int, default=0, help='5g: named export / splicing genes per panel (0 = all)')
    ap.add_argument('--d-highlight', required=True, help='data/fig5d_volcano_highlight_genes.csv (written by 5d)')
    ap.add_argument('--deg-lfc', type=float, default=2.0, help='bulk DEG = padj < .05 and |log2FC| >= this (as 5d)')
    ap.add_argument('--splice-k', type=int, default=20, help='5k: ExPert top-k tested at family level')
    ap.add_argument('--n-perm', type=int, default=5000, help='5k: random gene sets for the null')
    ap.add_argument('--dpsi-cut', type=float, default=0.1, help='5k: significant splicing = FDR < .05 and |dPSI| >= this')
    return ap


if __name__ == '__main__':
    main(parser().parse_args())
