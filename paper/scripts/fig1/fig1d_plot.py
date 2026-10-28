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

"""Fig. 1d: learned prototype geometry and its agreement with priors. Reads fig1d_data.py output.

    map         cosine UMAP of the atoms, coloured by named Leiden cluster (--clustering; the rest
                grey), each cluster labelled at its densest point.
    agreement   per prior, kNN overlap enrichment and pairwise-cosine Spearman against the
                atoms; filled = all atoms, open = atoms the prior term never pulled; grey band =
                label-shuffled prior (mean +/- sd). Priors the model trained on are marked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
# SVG text as real <text> objects (editable), not glyph outlines; PDF text as TrueType
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
from matplotlib.gridspec import GridSpec  # noqa: E402

SHORT = {'Processing of Capped Intron-Containing Pre-mRNA': 'pre-mRNA splicing',
         'GTP hydrolysis and joining of the 60S ribosomal subunit': 'cytosolic ribosome',
         'L13a-mediated translational silencing of Ceruloplasmin expression':
             'cytosolic ribosome',
         'Mitochondrial translation': 'mitochondrial translation',
         'rRNA processing in the nucleus and cytosol': 'rRNA processing',
         'M Phase': 'mitosis (M phase)',
         'RNA Polymerase II Pre-transcription Events': 'Pol II transcription initiation',
         'Proteasome assembly': 'proteasome',
         'Activation of the pre-replicative complex': 'DNA replication licensing',
         'Asparagine N-linked glycosylation': 'N-linked glycosylation',
         'Expression and translocation of olfactory receptors': 'olfactory receptors'}
PRIOR_LABEL = {'genomic position': 'genomic neighbours'}


# functional themes for clade colouring: first matching keyword wins (order matters)
THEMES = [
    ('mitochondria / OXPHOS', ('mitochondri', 'respiratory', 'electron transport', 'oxidative phos',
                               'atp synthesis')),
    ('ribosome biogenesis', ('rrna', 'ribosome biogenesis', 'subunit biogenesis', 'snorna',
                             'ribosomal large subunit biogenesis')),
    ('cytosolic translation', ('cytoplasmic translation', 'translation', 'ribosom', 'trna',
                               'aminoacyl', 'l13a', 'selenocysteine')),
    ('mRNA processing', ('splic', 'mrna', 'snrna', 'p-body', 'spliceosom')),
    ('transcription', ('transcription', 'rna polymerase', 'mediator', 'preinitiation')),
    ('chromatin', ('histone', 'chromatin', 'acetyl', 'hdac', 'methyltransferase complex')),
    ('DNA replication / mitosis', ('dna replication', 'dna metabolic', 'telomere', 'dna repair',
                                   'mitotic', 'm phase', 'cell cycle', 'centrosome', 'prometaphase',
                                   'chromosome segregation', 'kinetochore', 'spindle',
                                   'replication')),
    ('proteostasis', ('proteasom', 'protein folding', 'neddylation', 'ubiquitin', 'sumo',
                      'chaperon', 'unfolded protein')),
    ('secretion / glycosylation', ('golgi', 'vesicle', 'glycosyl', 'n-glycan', 'endoplasmic',
                                   'er to golgi', 'copii', 'copi', 'secret')),
]
BROAD = ('infection', 'disease', 'cancer', 'carcinoma', 'leukemia', 'virus', 'syndrome')


def clean_term(t: str) -> str:
    import re
    t = re.sub(r'\s*\(GO:\d+\)', '', str(t))
    t = re.sub(r'\s*R-HSA-\d+', '', t)
    t = t.replace(' (human)', '')
    return t if len(t) <= 36 else t[:34].rstrip() + '…'


def theme_of(term) -> str | None:
    if not isinstance(term, str):
        return None
    tl = term.lower()
    if any(b in tl for b in BROAD):
        return None
    for name, keys in THEMES:
        if any(k in tl for k in keys):
            return name
    return 'other'


LABEL_SHORT = {
    'mRNA Splicing, Via Spliceosome': 'spliceosome',
    'GTP Hydrolysis And Joining Of 60S Ribosomal Subunit': 'cytosolic ribosome',
    'M Phase': 'mitosis (M phase)',
    'Respiratory Electron Transport, ATP Synthesis By Chemiosmotic Coupling, Heat Production By '
    'Uncoupling Proteins': 'respiratory chain / ATP synthase',
    'Mitochondrial Translation': 'mitochondrial translation',
    'DNA Metabolic Process': 'DNA replication / repair',
    'Transcription Initiation At RNA Polymerase II Promoter': 'Pol II transcription initiation',
    'rRNA Processing In Nucleus And Cytosol': 'rRNA processing',
    'Proteasome': 'proteasome',
    'Transport Of Mature mRNA Derived From An Intron-Containing Transcript': 'mRNA export',
    'Ribosomal Large Subunit Biogenesis': '60S biogenesis',
    'Golgi Vesicle Transport': 'Golgi vesicle transport',
    'Positive Regulation Of Stem Cell Population Maintenance': 'stem cell maintenance',
    'Regulation Of Nucleic Acid-Templated Transcription': 'transcription regulation',
    'Positive Regulation Of Nucleic Acid-Templated Transcription': 'transcription activation',
    'RNA Polymerase III Transcription Initiation From Type 2 Promoter': 'Pol III transcription',
    'Formation Of RNA Pol II Elongation Complex': 'Pol II elongation',
    'Protein processing in endoplasmic reticulum': 'ER protein processing',
    '28S ribosomal subunit, mitochondrial': 'mitoribosome (28S)',
    'Protein Targeting To Mitochondrion': 'mitochondrial protein import',
    'Regulation Of Cell Fate Specification': 'cell fate specification',
    'Processing And Activation Of SUMO': 'SUMO processing',
    'Transcriptional Regulation By E2F6': 'E2F6 targets',
    'Regulation Of DNA Repair': 'DNA repair regulation',
    'Endoplasmic Reticulum Membrane': 'ER membrane',
    'Adaptive Immune System': 'adaptive immunity',
    'Multivesicular Body Assembly': 'multivesicular body assembly',
    "mRNA 3'-End Processing": "mRNA 3'-end processing",
    'Chromatin Modifying Enzymes': 'chromatin modifiers',
    'UV Response Dn': 'UV response (down)',
}


KEEP_CASE = {'Golgi', 'Ras', 'ER', 'DNA', 'RNA', 'SUMO', 'SAGA', 'UV', 'NF-kappa', 'B', 'DAP12',
             'E2F6', 'Pol', 'II', 'III', 'M', 'P-body', 'mRNA', 'rRNA', 'tRNA', 'mTOR'}


def sentence_case(t: str) -> str:
    """'Bile Acid Metabolism' -> 'bile acid metabolism'; acronyms / gene and complex names kept."""
    out = []
    for w in t.split(' '):
        keep = (w in KEEP_CASE or any(ch.isdigit() for ch in w) or w.isupper()
                or any(ch.isupper() for ch in w[1:]))
        out.append(w if keep else w.lower())
    return ' '.join(out)


def short_label(term: str) -> str:
    import re
    t = re.sub(r'\s*\(GO:\d+\)', '', str(term))
    t = re.sub(r'\s*R-HSA-\d+', '', t).replace(' (human)', '')
    return sentence_case(LABEL_SHORT.get(t, clean_term(t)))


def distinct_palette(n: int) -> list:
    """n categorical colours from scanpy's 102-colour palette (godsnot_102), dropping its
    near-grey / near-black / near-white entries so no class reads as the grey 'unannotated'."""
    from matplotlib.colors import to_rgb, rgb_to_hsv
    from scanpy.plotting.palettes import default_102
    keep = []
    for h in default_102:
        r = np.array(to_rgb(h))
        _, sat, val = rgb_to_hsv(r)
        lum = 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]
        # labels are drawn in the cluster colour, so the colour must also read as TEXT on white
        if sat >= 0.25 and 0.25 <= val and not (sat < 0.35 and val > 0.9) and lum <= 0.55:
            keep.append(h)
    if n > len(keep):
        raise ValueError(f'{n} classes but only {len(keep)} usable colours in default_102')
    return keep[:n]


MAX_SPECIFIC_SET = 300   # a term annotating more prototypes than this is too broad to name a cluster


def select_cluster_terms(d: Path, clustering: str, src_dir=None) -> pd.DataFrame:
    """One naming term per cluster, or none.

    A term is SPECIFIC if it is not an infection/disease term (theme_of) and annotates at most
    MAX_SPECIFIC_SET prototypes in its library. A cluster whose strongest term is an
    infection/disease term is not named at all. Every candidate already passed the enrichment
    criteria (BH q < 0.05, covers >= 20% of the cluster's annotated members). Clusters are named in
    order of their best specific term's significance; a cluster whose label is already taken
    falls back to its next specific term. Clusters with no specific term are left unnamed (grey).
    Writes fig1d_cluster_labels.csv (to src_dir, default d)."""
    terms = pd.read_csv(d / 'fig1d_clade_terms.csv')
    terms = terms[terms.clustering == clustering].sort_values('p')
    # a cluster whose STRONGEST term is an infection/disease set (in practice the zinc-finger
    # 'herpes simplex virus 1 infection' clusters) stays unnamed: weaker secondary terms must not
    # rescue it
    best = terms.groupby('cluster').head(1).set_index('cluster').term
    ok_cl = best.index[[theme_of(t) is not None for t in best]]
    terms = terms[terms.cluster.isin(ok_cl)
                  & np.array([theme_of(t) is not None for t in terms.term])
                  & (terms.set_size <= MAX_SPECIFIC_SET)]
    first = terms.groupby('cluster').head(1).sort_values('p')
    used, rows = set(), []
    for c in first.cluster:
        for _, r in terms[terms.cluster == c].iterrows():
            k = _norm_label(short_label(r.term))
            if k not in used:
                used.add(k)
                rows.append({**r.to_dict(), 'label': short_label(r.term),
                             'best_specific_term': first.loc[first.cluster == c, 'term'].iat[0]})
                break
    sel = pd.DataFrame(rows).set_index('cluster')
    sel['fallback'] = sel.term != sel.best_specific_term
    sel.drop(columns='clustering').to_csv(Path(src_dir or d) / 'fig1d_cluster_labels.csv')
    return sel


def annot_info(d: Path, lay: pd.DataFrame):
    """Per-prototype neighbourhood-enrichment annotation (fig1d_annotate.py), in the same shape
    as cluster_info: lay['cl'] = assigned term label (NaN = unannotated), spec indexed by label
    (largest group first), one colour per label."""
    an = pd.read_parquet(d / 'fig1d_annot_prototypes.parquet').set_index('perturbation')
    lay = lay.assign(cl=an.label.reindex(lay.perturbation).to_numpy())
    cnt = an.label.value_counts()
    spec = pd.DataFrame({'label': cnt.index, 'n': cnt.to_numpy()}, index=cnt.index)
    return lay, spec, dict(zip(spec.index, distinct_palette(len(spec))))


def cluster_info(d: Path, lay: pd.DataFrame, clustering: str, src_dir=None):
    """(lay with cl, named clusters sorted by p, colour per cluster)."""
    clc = pd.read_parquet(d / 'fig1d_tree_clusters.parquet').set_index('perturbation')
    lay = lay.assign(cl=clc[clustering].reindex(lay.perturbation).to_numpy())
    spec = select_cluster_terms(d, clustering, src_dir).sort_values('p')
    return lay, spec, dict(zip(spec.index, distinct_palette(len(spec))))

def _norm_label(lab: str) -> str:
    """Comparison key: near-identical names ('neddylation' / 'protein neddylation',
    'transcription regulation' / 'regulation of transcription') collide."""
    import re
    t = lab.lower()
    t = re.sub(r'\b(positive|negative)\s+regulation\s+of\b', ' ', t)
    t = re.sub(r'\b(regulation|regulation of|protein|pathway|signaling|process|the)\b', ' ', t)
    return ' '.join(sorted(t.split()))


def densest(X: np.ndarray) -> np.ndarray:
    """The point with the smallest distance to its 10th same-cluster neighbour."""
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return X[np.argmin(np.sort(D, 1)[:, min(10, len(X) - 1)])]


def umap_panel(d: Path, lay: pd.DataFrame, clustering: str, vs, out_dir, name: str, src_dir=None) -> None:
    """Fig. 1d: UMAP of the prototypes; every specifically enriched cluster has its own colour and
    a label placed next to it (short leader line where the label had to move). Grey = broad or
    unannotated clusters. No legend, title or notes."""
    import matplotlib.patheffects as pe
    from adjustText import adjust_text
    lay, spec, ccol = (annot_info(d, lay) if clustering == 'annot'
                       else cluster_info(d, lay, clustering, src_dir))
    labels = spec.label.to_dict()
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    rest = ~lay.cl.isin(spec.index)
    # VECTOR dots (not rasterized): the svg must stay editable point by point
    ax.scatter(lay.umap1[rest], lay.umap2[rest], s=5, color=vs.EXCLUDED, linewidths=0,
               rasterized=False, gid='grey_unannotated')
    for c in spec.index[::-1]:
        m = lay.cl == c
        ax.scatter(lay.umap1[m], lay.umap2[m], s=7, color=ccol[c], linewidths=0, alpha=0.9,
                   rasterized=False, gid=f'cluster_{c}_{labels[c]}')
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.grid(False)
    P = lay[['umap1', 'umap2']].to_numpy()
    span = np.ptp(P, axis=0)
    ctr = P.mean(0)
    texts, anchors = [], []
    for c, r in spec.iterrows():
        X = lay.loc[(lay.cl == c).to_numpy(), ['umap1', 'umap2']].to_numpy()
        a0 = densest(X)
        v = a0 - ctr
        v = v / (np.linalg.norm(v / span) + 1e-9)        # unit step in span-normalised units
        start = a0 + 0.035 * v                             # just outside the cluster, outward
        anchors.append(a0)
        texts.append(ax.text(start[0], start[1], labels[c], fontsize=vs.FONT_MIN - 3,
                             color=ccol[c], ha='center', va='center', zorder=5,
                             fontweight='bold',
                             # a light box, not a stroke halo: path effects make the SVG draw text as outlines
                             bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none', alpha=0.7)))
    pad = 0.06 * span
    ax.set_xlim(P[:, 0].min() - pad[0], P[:, 0].max() + pad[0])
    ax.set_ylim(P[:, 1].min() - pad[1], P[:, 1].max() + pad[1])
    rng = np.random.default_rng(0)
    obst = P[rng.choice(len(P), size=min(2500, len(P)), replace=False)]
    anchors = np.asarray(anchors)
    adjust_text(texts, x=np.r_[anchors[:, 0], obst[:, 0]], y=np.r_[anchors[:, 1], obst[:, 1]],
                target_x=anchors[:, 0], target_y=anchors[:, 1], ax=ax, ensure_inside_axes=True,
                expand=(1.08, 1.25), force_text=(0.4, 0.7), force_static=(0.15, 0.3),
                time_lim=60)
    # leader lines in the cluster's colour, drawn under the (white-haloed) label; only where the
    # label actually moved away from its cluster
    for t, a0, c in zip(texts, anchors, spec.index):
        x1, y1 = t.get_position()
        if np.hypot(*((np.array([x1, y1]) - a0) / span)) > 0.02:
            ax.plot([a0[0], x1], [a0[1], y1], color=ccol[c], lw=0.8, zorder=4)
    # axis glyph at the lower-left corner
    xa, ya = P[:, 0].min() - 0.5 * pad[0], P[:, 1].min() - 0.5 * pad[1]
    L = 0.07 * span.max()
    kw = dict(arrowstyle='-|>', color=vs.INK2, lw=1.2, mutation_scale=12)
    ax.annotate('', xy=(xa + L, ya), xytext=(xa, ya), arrowprops=kw)
    ax.annotate('', xy=(xa, ya + L), xytext=(xa, ya), arrowprops=kw)
    ax.text(xa + L / 2, ya - 0.2 * L, 'UMAP 1', ha='center', va='top', color=vs.INK2,
            fontsize=vs.FONT_MIN - 2)
    ax.text(xa - 0.2 * L, ya + L / 2, 'UMAP 2', ha='right', va='center', rotation=90,
            color=vs.INK2, fontsize=vs.FONT_MIN - 2)
    vs.save(fig, out_dir, name)
    print(f'[1d] {len(spec)} coloured + labelled clusters; grey {int(rest.sum()):,}')


def agreement_panel(agr: pd.DataFrame, meta: dict, vs, out_dir, name: str) -> None:
    """Fig. 1e: per prior, kNN overlap / chance and pairwise-cosine Spearman, all vs not-pulled."""
    fig = plt.figure(figsize=(8, 11))
    gsr = GridSpec(1, 2, figure=fig, width_ratios=[3.0, 2.4], wspace=0.08, bottom=0.15,
                   top=0.985, left=0.27, right=0.97)
    ax_k = fig.add_subplot(gsr[0, 0])
    ax_r = fig.add_subplot(gsr[0, 1], sharey=ax_k)
    order = (agr[(agr.subset == 'all') & (agr.metric == 'knn_enrich')]
             .sort_values('value').prior.tolist())
    y = {p: i for i, p in enumerate(order)}
    trained = set(agr.loc[agr.used_in_training, 'prior'])
    OFF = 0.09                                        # all / not-pulled sit close together
    for axx, met, xl in ((ax_k, 'knn_enrich', f"kNN overlap / chance  (k = {meta['k']})"),
                         (ax_r, 'rsa_spearman', 'pairwise-cosine\nSpearman ρ')):
        for _, r in agr[agr.metric == met].iterrows():
            yy = y[r.prior] + (OFF if r.subset == 'all' else -OFF)
            axx.plot([r.null_mean - 2 * r.null_sd, r.null_mean + 2 * r.null_sd], [yy, yy],
                     color=vs.EXCLUDED, lw=5, solid_capstyle='butt', zorder=1)
            axx.plot([r.value], [yy], 'o', ms=8, zorder=3,
                     mfc=vs.C['blue'] if r.subset == 'all' else 'white',
                     mec=vs.C['blue'], mew=1.8)
        axx.set_xlabel(xl)
        axx.xaxis.grid(True, color=vs.GRID); axx.yaxis.grid(False)
        axx.axvline(1.0 if met == 'knn_enrich' else 0.0, color=vs.MUTED, lw=1, zorder=0)
    ax_k.set_xscale('log')
    ax_k.set_xticks([1, 3, 10, 30, 100]); ax_k.set_xticklabels(['1', '3', '10', '30', '100'])
    ax_k.set_yticks(range(len(order)))
    ax_k.set_yticklabels([PRIOR_LABEL.get(p, p) + (' *' if p in trained else '') for p in order])
    ax_r.tick_params(labelleft=False)
    ax_k.set_ylim(-0.6, len(order) - 0.4)
    h = [plt.Line2D([], [], ls='', marker='o', ms=8, mfc=vs.C['blue'], mec=vs.C['blue']),
         plt.Line2D([], [], ls='', marker='o', ms=8, mfc='white', mec=vs.C['blue'], mew=1.8),
         plt.Line2D([], [], color=vs.EXCLUDED, lw=5), plt.Line2D([], [], ls='')]
    fig.legend(h, ['all prototypes', 'not prior-pulled', 'shuffled prior (±2 sd)',
                   '* training prior'], loc='lower right', bbox_to_anchor=(0.98, 0.0), ncol=2,
               frameon=False, fontsize=vs.FONT_MIN - 2, handlelength=1.4, columnspacing=1.0)
    vs.save(fig, out_dir, name)


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'                              # paper figures: plain white background
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'})


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d = Path(a.data_dir)
    lay = pd.read_parquet(d / 'fig1d_layout.parquet')
    agr = pd.read_csv(d / 'fig1e_agreement.csv')
    meta = json.load(open(d / 'fig1e_meta.json'))
    src = 'annot' if a.colour_by == 'annot' else a.clustering
    umap_panel(d, lay, src, vs, a.out_dir, a.name_umap)
    agreement_panel(agr, meta, vs, a.out_dir, a.name_agreement)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--name-umap', default='fig1d_prototype_umap')
    ap.add_argument('--name-agreement', default='fig1e_prior_agreement')
    ap.add_argument('--clustering', default='leiden_4', help='column of fig1d_tree_clusters.parquet')
    ap.add_argument('--colour-by', choices=['annot', 'cluster'], default='cluster',
                    help='annot = per-prototype neighbourhood enrichment; cluster = named Leiden clusters')
    main(ap.parse_args())
