# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Write a figure notebook from a list of cells, with the standard setup cell first.

    from nbbuild import build
    build('paper/figures/si/figS14/figS14.ipynb', '# Figure S14 ...', [
        ('code', 'import figS14_plot as P\\n...'),
        ('md', '## A ...'),
        ('code', 'P.panel_a(DATA, OUT, vs, meta)\\nnb.show(OUT, "figS14a_celltypes")'),
    ])

Setup defines nb (nbtools), vs (vizstyle) and DATA / INPUTS / OUT / SRC (data/, inputs/, render/, render/src/).
"""
from __future__ import annotations

import nbformat

SETUP = """import sys
from pathlib import Path
_paper = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts' / 'common').is_dir())
sys.path.insert(0, str(_paper / 'scripts' / 'common'))
import nbtools as nb                      # puts paper/scripts/*/ on sys.path

vs = nb.vizstyle()
DATA, INPUTS, OUT, SRC = nb.folders()     # data/ (cache), inputs/, render/, render/src/"""


def build(path, title_md, cells):
    nbk = nbformat.v4.new_notebook()
    nbk.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbk.cells = [nbformat.v4.new_markdown_cell(title_md), nbformat.v4.new_code_cell(SETUP)]
    for kind, src in cells:
        nbk.cells.append(nbformat.v4.new_markdown_cell(src) if kind == 'md' else nbformat.v4.new_code_cell(src))
    nbformat.write(nbk, path)
