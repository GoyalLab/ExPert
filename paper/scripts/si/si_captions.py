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

"""Concatenate every SI legend into one file: the figure captions (--si-dir/figSN/caption.md, in figure-number
order) followed by the table legends (--si-dir/tables/captions.md). The per-figure files stay the source; rerun
this after editing any of them. Writes --out.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def body(p):
    """Caption text without the leading licence comment."""
    return re.sub(r'^\s*<!--.*?-->\s*', '', p.read_text(), count=1, flags=re.S).strip()


def main(a):
    si = Path(a.si_dir)
    figs = sorted(si.glob('figS*/caption.md'), key=lambda p: int(re.search(r'figS(\d+)', str(p)).group(1)))
    head = re.match(r'^\s*<!--.*?-->', (si / 'tables' / 'captions.md').read_text(), flags=re.S).group(0)
    parts = [head, '# Supplementary figures'] + [body(p) for p in figs]
    parts += ['# Supplementary tables', body(si / 'tables' / 'captions.md')]
    Path(a.out).write_text('\n\n'.join(parts) + '\n')
    print(f'{len(figs)} figure captions + table legends -> {a.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--si-dir', required=True, help='paper/figures/si')
    ap.add_argument('--out', required=True, help='paper/figures/si/captions.md')
    main(ap.parse_args())
