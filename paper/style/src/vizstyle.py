"""Shared figure style for the ExPert-data result plots.

One validated categorical palette, applied in fixed slot order and never cycled, so that
a series keeps its colour across every figure in the paper.  The palette passes the
colour-vision-deficiency and normal-vision separation checks on the light surface used
here; three of the slots sit below 3:1 contrast against that surface, so identity must
always be carried by a legend or a direct label, never by hue alone.
"""
from __future__ import annotations

import matplotlib

# Fixed categorical order.  Never reassign by rank -- colour follows the entity.
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
     'magenta': '#e87ba4', 'green': '#008300', 'violet': '#4a3aa7', 'red': '#e34948'}
ORDER = [C['blue'], C['orange'], C['aqua'], C['yellow'],
         C['magenta'], C['green'], C['violet'], C['red']]

# Ten-slot extension, for categorical variables that genuinely have 9-10 levels (the
# corpus has 10 distinct cell lines).  The eight above plus a cyan and a magenta-maroon,
# in this order.  Verified with the palette checker: ALL CHECKS PASS on the light surface
# -- worst adjacent CVD dE 9.1 (protan), worst adjacent normal-vision dE 19.6.  Standard
# 10-colour palettes were tried first and all failed: Tol muted, Okabe-Ito extended and
# tab10 each break the chroma floor or drop adjacent CVD dE below 8.  Do not reorder --
# the checks are on ADJACENT pairs, so the sequence is part of what passes.
CYAN, MAROON = '#0e9aa7', '#a82359'
ORDER10 = ORDER + [CYAN, MAROON]
# single-hue sequential ramp, light -> dark (magnitude encodings only)
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
       '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8983'
SURFACE, GRID = '#ffffff', '#e5e5e5'   # pure white surface (paper rule); neutral grey grid
# "included vs not" uses green/grey rather than two hues: the second class is an absence,
# and a neutral reads as such where a second hue would imply a second category.
INCLUDED, EXCLUDED = '#1baf7a', '#d6d5d0'

# 14 pt is the floor for every text element -- these figures are read at print size in a
# paper, where anything smaller is illegible.  Nothing below FONT_MIN is set anywhere.
FONT_MIN = 14

RC = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': FONT_MIN, 'axes.titlesize': FONT_MIN + 2, 'axes.labelsize': FONT_MIN,
    'xtick.labelsize': FONT_MIN, 'ytick.labelsize': FONT_MIN,
    'legend.fontsize': FONT_MIN, 'figure.titlesize': FONT_MIN + 3,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.autolayout': False,
    # text stays editable text: SVG <text> (not glyph outlines), PDF TrueType (not Type 3)
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
}


def apply() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)


def seq_cmap():
    return matplotlib.colors.LinearSegmentedColormap.from_list('expert_seq', SEQ)


def vectorize(fig) -> None:
    """Turn off rasterization on every artist, so the SVG holds individual vector marks.

    Catches rasterized=True scatters AND colorbar solids, which matplotlib rasterizes on its
    own once a colorbar has >= 50 colour steps. imshow stays a bitmap in any backend, so
    heatmaps must be drawn with pcolormesh.
    """
    for o in fig.findobj():
        if o.get_rasterized():
            o.set_rasterized(False)
    plain_log_ticks(fig)


_SUP = str.maketrans('-0123456789.', '\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u22c5')


def _pow10(v, _pos=None) -> str:
    """10^k tick label in Unicode (10\u2075), never mathtext; 0 and non-decades fall back to plain numbers."""
    import math
    if v == 0:
        return '0'
    e = math.log10(abs(v))
    if abs(e - round(e)) > 1e-9:
        return ''                                     # minor / non-decade ticks stay unlabelled
    k = int(round(e))
    lab = '1' if k == 0 else ('10' if k == 1 else '10' + str(k).translate(_SUP))
    return ('\u2212' if v < 0 else '') + lab


def plain_log_ticks(fig) -> None:
    """Replace matplotlib's default log / symlog tick labels (always mathtext, i.e. one SVG glyph per
    character) with Unicode ones, so every tick label is a single editable text object. Custom
    formatters a script set itself are left alone."""
    from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() in ('log', 'symlog') and \
                    isinstance(axis.get_major_formatter(), LogFormatterSciNotation):
                axis.set_major_formatter(FuncFormatter(_pow10))
            if axis.get_scale() in ('log', 'symlog') and \
                    isinstance(axis.get_minor_formatter(), LogFormatterSciNotation):
                axis.set_minor_formatter(FuncFormatter(lambda v, p=None: ''))


def heatmap(ax, C, **kw):
    """Vector stand-in for ax.imshow(C, aspect='auto'): one quad per cell, cells centred on
    integer coordinates, row 0 at the top. Masked cells take the colormap's bad colour."""
    import numpy as np
    m, n = np.shape(C)[:2]
    kw.setdefault('zorder', 0)                       # where imshow sits (under an axisbelow grid)
    mesh = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(m + 1) - 0.5, C, shading='flat',
                         edgecolors='face', linewidth=0.3, **kw)   # face edges hide hairline seams
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    return mesh


def assert_vector_svg(path) -> None:
    """Raise if the written SVG embeds a raster <image>, draws its text as glyph outlines, or splits a label into
    per-glyph positioned pieces (paper rules: no rasterized plots; every label one editable text object --
    needs rcParams['svg.fonttype'] = 'none', set by apply(), and no mathtext)."""
    import re
    from pathlib import Path
    t = Path(path).read_text()
    if '<image' in t:
        raise ValueError(f'{path} embeds a raster <image>; draw heatmaps with pcolormesh, not imshow')
    if re.search(r'<path id="[A-Za-z][\w-]*-[0-9a-f]+"', t):    # glyph defs, incl. DejaVuSans-Bold-xx
        raise ValueError(f"{path} draws text as glyph outlines; set svg.fonttype='none' (vizstyle.apply()) "
                         f"and do not use path_effects on text (a halo outlines it)")
    # path_effects (e.g. a white stroke halo) write the label as an anonymous <path> inside its text group
    # (parsed, not regexed: a text group with a bbox nests a patch group before its <text>)
    import xml.etree.ElementTree as ET
    ns = '{http://www.w3.org/2000/svg}'
    for g in ET.fromstring(t).iter(f'{ns}g'):
        if re.fullmatch(r'text_\d+', g.get('id', '')) and g.find(f'.//{ns}path') is not None \
                and g.find(f'.//{ns}text') is None:
            raise ValueError(f'{path} draws a label as an outline (path_effects on text?); '
                             f'use a bbox instead of a stroke halo')
    # mathtext ($...$) is written as <tspan>s with ONE x per glyph, which Illustrator imports as separate
    # letters: use Unicode (CD4⁺, ×, 10³) instead so every label is a single text object
    if re.search(r'<tspan[^>]* x="-?[\d.]+ -?[\d.]+', t):
        raise ValueError(f'{path} has per-glyph positioned text (mathtext); use Unicode instead of $...$')


def save(fig, out_dir, name: str, dpi: int = 300) -> None:
    """Write SVG + high-resolution PNG side by side; the SVG is fully vector."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorize(fig)
    for ext, kw in (('svg', {}), ('png', {'dpi': dpi})):
        fig.savefig(out_dir / f'{name}.{ext}', bbox_inches='tight', **kw)
    assert_vector_svg(out_dir / f'{name}.svg')
    plt.close(fig)
    print(f'  wrote {name}.svg / .png', flush=True)


def thousands(x, _pos=None) -> str:
    return f'{int(x):,}'
