# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Supplementary Fig. S7: Figure 4B-E redrawn with EFFGATE3 (no K562 in training) as ExPert.

Render-only (login is fine). Uses fig4_plot's own panel functions on figS7_data.py's tables, so the
panels are drawn exactly like Figure 4 and differ only in the model behind the ExPert line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig4_plot as F4  # noqa: E402

def _lighten(hexc, f):
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*(round(c + (255 - c) * f) for c in (r, g, b)))


LIGHT = _lighten(F4.EXPERT, 0.35)
NAMES = {"fig4b_sciplex_methods": "figS7a_sciplex_methods", "fig4c_sciplex_dose": "figS7b_sciplex_dose",
         "fig4d_hdac_machinery": "figS7c_hdac_machinery", "fig4e_ar42": "figS7d_ar42"}


_BASE_STYLE = F4.method_style
PANELS = {"a": F4.panel_b_sciplex, "b": F4.panel_c_sciplex, "c": F4.panel_d_sciplex, "d": F4.panel_e_ar42}


def style(vs):
    vs.apply()
    F4.VS = vs                                   # fig4_plot's drawing reads its module-level VS
    F4.SPB = "."                                 # panel functions read d / SPB / <table>
    # the K562-free model in a slightly lighter ExPert orange (35% toward white), so it reads as ExPert
    # but is not mistaken for the shipped model; fig4_plot's own EXPERT_LIGHT is too pale for a line
    F4.method_style = lambda: {**_BASE_STYLE(), "ExPert - no K562": (LIGHT, "o", 2.6)}


def panel(key, d, out, vs):
    """Draw one panel with fig4_plot's function, then rename its fig4* files to the figS7* name."""
    fn = PANELS[key]
    old = {F4.panel_b_sciplex: "fig4b_sciplex_methods", F4.panel_c_sciplex: "fig4c_sciplex_dose",
           F4.panel_d_sciplex: "fig4d_hdac_machinery", F4.panel_e_ar42: "fig4e_ar42"}[fn]
    new = NAMES[old]
    fn(d, out)
    for ext in ("svg", "png"):
        p = out / f"{old}.{ext}"
        if p.exists():
            p.replace(out / f"{new}.{ext}")
    vs.assert_vector_svg(out / f"{new}.svg")
    print(f"  {new}.svg / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vizstyle-root", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in PANELS:
        panel(k, d, out, vs)


if __name__ == "__main__":
    main()
