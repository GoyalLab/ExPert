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

"""Table S7: the prior screen (deployment panel, full 19,895-row vocabulary; related to Fig. 3), read through the
screen's own loader (paper/analysis/readout/prior_screen/plot.py) and Fig. 3's single-prior table, so every number matches
the screen's REPORT.md. Zero-shot = effect-size-weighted top-10 recall of the unseen perturbations, x chance.
Written to --out-dir (si/tables) as Table_S7.xlsx (one sheet per part) plus one CSV per part.

    S7A single priors      every single prior: family, KEGG neighbour lift (size-matched), zero-shot, hits, shuffled null
    S7B prior subsets      every subset of the ten best single priors: members, size, zero-shot overall and per random
                           half of the perturbations, shuffled-gene twin, number of text priors, whether a source family
                           appears twice, rank, and whether it beats the production trio
    S7C production prior   the shipped text prior (Opus 4.8) vs Opus 4.7 (both qwen3-embed-8b): alone and in the trio
                           with DepMap and STRING

The production text prior fell out of the ten-best pool under the cos^4 fill, so its trio is not among the subsets: both
trios were scored separately on the same screen (screen.py panels, --trio-log) and are read from that log's full-panel
block. 'beats_production' compares every subset against that score.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

G = 'pooled_all'                                  # all unseen perturbations, pooled over screens
PRODUCTION = ('opus-4.8/qwen3-embed-8b', 'depmap_dstd', 'string')
ALT_TEXT = 'opus-4.7/qwen3-embed-8b'
TRIO_LABEL = {PRODUCTION[0]: 'production trio text(4.8)+dstd+string', ALT_TEXT: 'trio text(4.7)+dstd+string'}
SHEET = {'S7A': 'S7A single priors', 'S7B': 'S7B prior subsets', 'S7C': 'S7C production prior'}


def members(x) -> list[str]:
    return list(x) if isinstance(x, (list, tuple)) else [m.strip() for m in str(x).split('+')]


def trio_scores(log: str) -> dict[str, dict]:
    """the two trios from screen.py panels' full-vocabulary block: label, w_x, x, w_top10, top10, hits"""
    txt = Path(log).read_text()
    block = txt[txt.index('=== panel full'):]
    out = {}
    for t, lab in TRIO_LABEL.items():
        m = re.search(re.escape(lab) + r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)', block)
        assert m, f'{lab} not in the full-panel block of {log}'
        out[t] = dict(w_x=float(m[1]), hits=int(m[5]))
    return out


def main(a):
    sys.path.insert(0, str(Path(a.prior_screen_root)))
    import plot as PS                                         # the screen's loader
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    singles = pd.read_csv(a.singles)
    s4a = singles.rename(columns={'KEGG_lift10_matched': 'kegg_neighbour_lift', 'w_zs_x': 'zero_shot_x_chance',
                                  'hits10': 'zero_shot_hits_top10', 'null_mean': 'shuffled_null_x_chance'}) \
        .sort_values('zero_shot_x_chance', ascending=False).reset_index(drop=True)

    cb, pool = PS.load_combos(a.screen_dir)
    fam = {p: f for _, p, f, _ in pool}
    pri = cb.priors.map(members)
    fams = pri.map(lambda m: [fam[p] for p in m])
    s4b = pd.DataFrame({'priors': pri.map(' + '.join), 'families': fams.map(' + '.join), 'size': cb['size'],
                        'zero_shot_x_chance': cb[f'{G}.w_zs_x'], 'zero_shot_half_a': cb[f'{G}.w_halfA'],
                        'zero_shot_half_b': cb[f'{G}.w_halfB'], 'shuffled_twin_x_chance': cb[f'{G}.w_null_zs_x'],
                        'n_text_priors': fams.map(lambda f: f.count('text')),
                        'family_repeated': fams.map(lambda f: len(set(f)) < len(f))})
    trio = trio_scores(a.trio_log)
    x_prod = trio[PRODUCTION[0]]['w_x']
    s4b['beats_production'] = s4b.zero_shot_x_chance > x_prod
    s4b = s4b.sort_values('zero_shot_x_chance', ascending=False).reset_index(drop=True)
    s4b.insert(0, 'rank', range(1, len(s4b) + 1))

    one = s4a.set_index('prior')
    s4c = pd.DataFrame([dict(text_prior=t, shipped=t == PRODUCTION[0],
                             single_zero_shot_x_chance=one.loc[t, 'zero_shot_x_chance'],
                             single_rank=int((s4a.zero_shot_x_chance > one.loc[t, 'zero_shot_x_chance']).sum()) + 1,
                             single_kegg_neighbour_lift=one.loc[t, 'kegg_neighbour_lift'],
                             in_top10_pool=t in fam,
                             trio_zero_shot_x_chance=trio[t]['w_x'], trio_zero_shot_hits_top10=trio[t]['hits'])
                        for t in (PRODUCTION[0], ALT_TEXT)])

    # numbers quoted in the text
    b = s4b[s4b.beats_production]
    per_source = s4b[~s4b.family_repeated]
    print(f'[S7B] {len(s4b)} subsets; {len(b)} beat the production trio ({x_prod:.2f}x); '
          f'{int(b.family_repeated.sum())} ({b.family_repeated.mean():.0%}) repeat a source family; '
          f'by text priors: {b[b.family_repeated].n_text_priors.value_counts().sort_index().to_dict()}', flush=True)
    print(f'[S7B] one prior per source: {len(per_source)} subsets, best {per_source.zero_shot_x_chance.iloc[0]:.2f}x '
          f'({per_source.families.iloc[0]}); production ranks '
          f'{int((per_source.zero_shot_x_chance > x_prod).sum()) + 1} of {len(per_source)}', flush=True)
    print(s4c.round(2).to_string(index=False), flush=True)

    with pd.ExcelWriter(out / 'Table_S7.xlsx') as w:
        for part, df in (('S7A', s4a), ('S7B', s4b), ('S7C', s4c)):
            df.to_excel(w, sheet_name=SHEET[part], index=False)
            df.to_csv(out / f'Table_{part}.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--singles', required=True, help='Fig. 3 data fig3f_source.csv (single priors)')
    ap.add_argument('--prior-screen-root', required=True, help='dir holding the screen plot.py')
    ap.add_argument('--screen-dir', required=True, help='full-vocabulary screen run dir (H4_full_p4, combos.json)')
    ap.add_argument('--trio-log', required=True, help='screen.py panels log scoring the two trios (panels_p4.sbatch)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
