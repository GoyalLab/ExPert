# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Non-human queries: detect a non-human gene space and move it onto human symbols.

ExPert's gene space is human HGNC symbols. A mouse query (``Actb``, ``Alb``, ``mt-Co1``) matches almost nothing
exactly but most of it once upper-cased, which is what `guess_species` looks for. `to_human` then maps it:

* with an ortholog table (MGI ``HOM_MouseHumanSequence.rpt``, columns ``DB Class Key``, ``NCBI Taxon ID``,
  ``Symbol``): homology classes with exactly ONE human gene only; several mouse genes of one class are SUMMED into
  that human gene (counts are additive). Mouse genes in classes with several human genes are dropped -- there is
  no honest way to split their counts.
* without one: upper-casing. Right for most one-to-one orthologs, wrong for renamed genes, so it is logged as a
  fallback.

`predict(orthologs='auto')` calls both; this module never touches a model.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

log = logging.getLogger(__name__)

_TITLE = re.compile(r'^[A-Z][a-z0-9]')            # Actb, Gm12345, Rpl13a -- mouse / rat style
_ENS_NONHUMAN = re.compile(r'^ENS[A-Z]{3,}G\d')   # ENSMUSG..., ENSRNOG... (human is ENSG...)


def guess_species(var_names, reference) -> dict:
    """Is this gene space human (matches `reference` as is) or a non-human symbol space (matches once upper-cased)?

    Returns a report dict with ``non_human`` (bool), the exact and upper-cased overlap with `reference` (fractions
    of the query genes), the share of title-case symbols and the share of non-human Ensembl IDs."""
    q = pd.Index(np.asarray(var_names).astype(str))
    ref = set(map(str, reference))
    n = max(len(q), 1)
    exact = float(q.isin(ref).sum()) / n
    upper = float(q.str.upper().isin(ref).sum()) / n
    title = float(sum(bool(_TITLE.match(g)) for g in q)) / n
    ens = float(sum(bool(_ENS_NONHUMAN.match(g)) for g in q)) / n
    # a human query matches as is; a mouse / rat symbol query gains most of its overlap from upper-casing
    non_human = (upper - exact > 0.2 and title > 0.5) or ens > 0.5
    return dict(non_human=bool(non_human), exact=round(exact, 4), upper=round(upper, 4), title_case=round(title, 4),
                ensembl_non_human=round(ens, 4), n_genes=int(len(q)))


def ortholog_map(var_names, table: str | pd.DataFrame, query_taxon: int = 10090,
                 target_taxon: int = 9606) -> pd.Series:
    """query symbol -> human symbol from an MGI-style homology table, one-to-one-in-human classes only."""
    H = pd.read_csv(table, sep='\t', low_memory=False) if not isinstance(table, pd.DataFrame) else table
    key = 'DB Class Key'
    q = H[H['NCBI Taxon ID'] == query_taxon][[key, 'Symbol']]
    h = H[H['NCBI Taxon ID'] == target_taxon][[key, 'Symbol']]
    n_h = h.groupby(key).size()
    m = q[q[key].isin(n_h[n_h == 1].index)].merge(h, on=key, suffixes=('_q', '_h'))
    m = m[m.Symbol_q.isin(set(map(str, var_names)))].drop_duplicates('Symbol_q')
    return pd.Series(m.Symbol_h.values, index=m.Symbol_q.values, name='human')


def to_human(adata: AnnData, table: str | pd.DataFrame | None = None, query_taxon: int = 10090) -> tuple[AnnData, dict]:
    """A copy of `adata` on human symbols (X summed per human gene; obs / obsm / uns kept) and a report dict.

    `table` = MGI-style homology table (path or DataFrame); ``None`` falls back to upper-casing."""
    names = np.asarray(adata.var_names).astype(str)
    if table is not None:
        mp = ortholog_map(names, table, query_taxon=query_taxon)
        method = 'homology table (one-to-one human classes, many-to-one summed)'
    else:
        mp = pd.Series([g.upper() for g in names], index=names)
        method = 'upper-case symbols (no ortholog table)'
    src = np.flatnonzero(pd.Index(names).isin(mp.index))
    tgt_names = mp.reindex(names[src]).to_numpy().astype(str)
    out_names, tgt = np.unique(tgt_names, return_inverse=True)
    # (query genes x human genes) 0/1 matrix: X @ M sums every query gene into its human gene
    M = sp.csr_matrix((np.ones(len(src), np.float32), (src, tgt)), shape=(len(names), len(out_names)))
    X = adata.X
    Xh = (X @ M) if sp.issparse(X) else np.asarray(X) @ M.toarray()
    Xh = sp.csr_matrix(Xh) if sp.issparse(X) else np.asarray(Xh)
    tot = float(X.sum()) if X.size else 0.0
    report = dict(method=method, n_query_genes=int(len(names)), n_mapped_query_genes=int(len(src)),
                  n_human_genes=int(len(out_names)), n_many_to_one=int(len(src) - len(out_names)),
                  frac_counts_kept=round(float(Xh.sum()) / tot, 4) if tot > 0 else float('nan'))
    new = AnnData(X=Xh, obs=adata.obs.copy(), var=pd.DataFrame(index=pd.Index(out_names, name=adata.var.index.name)),
                  uns=dict(adata.uns), obsm=dict(adata.obsm))
    new.uns['ortholog_conversion'] = report
    return new, report
