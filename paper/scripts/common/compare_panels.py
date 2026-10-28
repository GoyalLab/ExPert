#!/usr/bin/env python
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Compare the panels a notebook rendered (<fig-dir>/render/*.png) with the canonical ones (<fig-dir>/*.png).

Per panel: identical (same pixels), near (same size, < --tol of pixels differ by more than 8/255 in any channel),
differs, size (different pixel dimensions) or missing (listed in --panels but not rendered). Default panels: every
rendered PNG that has a canonical counterpart; canonical PNGs the notebook does not draw (hand-made schematics, photos,
alternative versions not in the figure) are listed as not_rendered. Written to <fig-dir>/render/compare.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def compare(a, b, tol):
    if not b.exists():
        return dict(status='missing')
    x = np.asarray(Image.open(a).convert('RGBA'), dtype=np.int16)
    y = np.asarray(Image.open(b).convert('RGBA'), dtype=np.int16)
    if x.shape != y.shape:
        return dict(status='size', canonical=f'{x.shape[1]}x{x.shape[0]}', rendered=f'{y.shape[1]}x{y.shape[0]}')
    d = np.abs(x - y).max(axis=2)
    frac = float((d > 8).mean())
    status = 'identical' if d.max() == 0 else 'near' if frac < tol else 'differs'
    return dict(status=status, frac_px_differ=round(frac, 6), max_diff=int(d.max()))


def main(a):
    fig = Path(a.fig_dir)
    out = fig / a.render
    canon = sorted(p.stem for p in fig.glob('*.png'))
    names = a.panels or [n for n in canon if (out / f'{n}.png').exists()]
    rows = [dict(panel=n, **compare(fig / f'{n}.png', out / f'{n}.png', a.tol)) for n in names]
    rows += [dict(panel=n, status='not_rendered') for n in canon if n not in names]
    t = pd.DataFrame(rows)
    t.to_csv(out / 'compare.csv', index=False)
    print(t.to_string(index=False))
    extra = sorted({p.stem for p in out.glob('*.png')} - set(names))
    if extra:
        print('rendered but not canonical:', extra)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig-dir', required=True)
    ap.add_argument('--render', default='render')
    ap.add_argument('--panels', nargs='*')
    ap.add_argument('--tol', type=float, default=1e-3)
    main(ap.parse_args())
