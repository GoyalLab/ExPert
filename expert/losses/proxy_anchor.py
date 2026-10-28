# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.
"""Proxy Anchor loss (Kim, Kim, Cho & Kwak, CVPR 2020) over the ATOM TABLE.

WHY THIS AND NOT THE CE WE ALREADY HAVE
---------------------------------------
The shipped arm (ad2) gives the atoms exactly ONE gradient: `_atom_anchor_loss`, an L2
regression of each atom onto the batch mean of its own class. `cls_align_weight` is 0, so
there is no inter-class competition on the atoms at all -- the only discriminative pressure
in the run reaches them second-hand, through a FREE classifier (`cls_head_w`) that reshapes
z. That is a per-class regression with no repulsion, and the offline dictionary fit says
what you would expect from one: the trained atoms (top1 0.1961) are a WORSE estimate of the
class direction than the plain class means they are regressing onto (0.2085), and worse
again than per-dataset pseudobulks (0.2261).

The obvious fix -- put cosine CE back on the atoms -- has been run and measured, and it
fails in a specific way: rank_med 39 vs 15 for the anchor-only arm, top-1 up and the tail
dead. The mechanism is visible in the gradient, and it is NOT that CE "never stops":

  SOFTMAX CE DIES ON THE SAMPLE. Its denominator contains the positive logit, so once
  p_true -> 1 the entire row's gradient vanishes REGARDLESS of how the remaining classes are
  ordered behind the winner. Measured in test_proxy_anchor at the real geometry (B=512,
  D=128, L=200): cosine CE at T=0.1 loses 386x of its atom gradient (ratio 2.6e-3) going
  from random to within-class-aligned. It optimises the argmax and then abandons the
  ordering, which is exactly the observed signature.

  THE PROXY ANCHOR NEGATIVE TERM IS A SOFTMAX OVER NEGATIVES ONLY.
  d/du_i log(1 + sum_j exp(u_j)) = exp(u_i) / (1 + sum_j exp(u_j)), and there is no positive
  logit anywhere in that denominator. So the repulsion does not switch off when the true
  class wins; it keeps redistributing pressure onto whichever negatives currently sit
  closest to the proxy. That is continuous hard-negative mining on the tail, and it is the
  actual reason to prefer this here. Measured: 0.51 of the atom gradient survives perfect
  within-class alignment, and all of the survivor is the negative half.

  THE POSITIVE HALF DOES SATURATE, and should. exp(-alpha*(s - delta)) decays once a class
  clears the margin (measured viol_pos 0.867 -> 0.000, margin -> 1.00), so the PULL stops
  while the PUSH continues. That asymmetry is the useful shape: the anchor and the positive
  term put an atom on its class-mean direction and then stop arguing about it, and only the
  repulsion keeps working.

WHAT delta AND THE "1 +" ACTUALLY DO IN OUR REGIME -- READ THIS BEFORE TUNING
----------------------------------------------------------------------------
The `1 +` is a null slot that absorbs the gradient only when EVERY negative is far below
-delta, i.e. sum_j exp(.) << 1. At 200 atoms in 128 dimensions that state is unreachable:
you cannot place 200 unit vectors in R^128 with all pairwise cosines below -0.1 (for L > D
you cannot even reach the -1/(L-1) simplex floor). Random 128-d directions sit at cosine
0 +/- 0.088, so alpha*(s + delta) is comfortably positive for the hardest negatives and the
sum runs in the thousands. CONSEQUENCE: delta is NOT an off switch for the repulsion here.
It sets the magnitude of a pressure that persists for the whole run, and the counterweight
that keeps atoms on the ADDITIVE class-mean direction (the one a group-OMP decode needs,
since a two-perturbation state is approximately a sum of class means and is not a sum of
maximally-discriminative directions) is the anchor's L2 pull, not this loss. Run them
together; do not expect delta to do the anchor's job.

alpha sets how concentrated the hard-negative mining is. At our cosine spread (sd 0.088)
alpha=32 puts most of the negative gradient on the single closest negative per proxy, which
is the noisiest possible target at ~3.4 cells per class -- treat it as the top of a ladder,
not a default to trust.

Relative to Proxy-NCA / ProxyNCA++ (Movshovitz-Attias 2017; Teh, DeVries & Taylor 2020):
those assign each SAMPLE to its proxy and normalise over proxies, so the denominator again
contains the positive and the sample-level death returns. Proxy Anchor inverts the roles --
the PROXY is the anchor and is associated with the whole batch -- which is what removes the
positive from the negative term's denominator and recovers data-to-data relations through
the shared proxy. That inversion is the reason to prefer it here.

A SECOND, INDEPENDENT REASON: EVERY PROXY IS UPDATED EVERY STEP. The negative term runs over
ALL live proxies, not only the classes that appear in the batch. At 200 classes / batch 512
the mean class occupancy is ~3.4 cells, so a batch-conditional term touches a minority of
atoms per step and the geometry it induces depends on batch composition. This one does not.

WHAT THIS TERM IS NOT
---------------------
It is not a replacement for the anchor. The anchor sets WHERE an atom sits and, being L2
rather than cosine, constrains its NORM -- which the group-OMP least-squares fit reads
directly as coefficient scale, and which every cosine metric is blind to. Proxy Anchor is
scale-free and only sets angular separation. Measured on the leaky ladder, anchor weight 0
was catastrophic (rank_med 28.0 vs 8.0 at 0.1), so the intended configuration is
anchor(low) + proxy anchor: reconstruction base + tail pressure, not one or the other.
`neg_only=True` exposes the minimal version -- keep the anchor's pull as the positive and add
ONLY the missing repulsion.

Defaults alpha=32, delta=0.1 are the paper's, kept so the module reproduces published
behaviour out of the box. Neither is defensible for this problem without a ladder -- see the
regime note above.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ['proxy_anchor_loss']


def _lse_with_zero(u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Column-wise log(1 + sum_{i in mask} exp(u_i)).

    Implemented as logsumexp over the sample axis with an explicit zero row prepended, which
    is the numerically stable spelling of the `1 +`. A column with no members returns exactly
    log(1) = 0, so an empty positive set is a no-op rather than a NaN.

    ``torch.finfo(dtype).min`` rather than ``-inf``: logsumexp of an all -inf column is NaN,
    and while the prepended zero row makes that unreachable here, the sentinel keeps the
    helper safe if it is ever reused without one.
    """
    u = u.masked_fill(~mask, torch.finfo(u.dtype).min)
    zero = torch.zeros(1, u.shape[1], device=u.device, dtype=u.dtype)
    return torch.logsumexp(torch.cat([zero, u], dim=0), dim=0)


def proxy_anchor_loss(
    z: torch.Tensor,
    atoms: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 32.0,
    delta: float = 0.1,
    neg_only: bool = False,
    z_grad: float = 1.0,
    relax_idx: torch.Tensor | None = None,
    relax_sim: torch.Tensor | None = None,
    relax_beta: float = 0.0,
    relax_sharpen: bool = False,
    valid: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    center: bool = True,
    center_atoms: bool = True,
    coh_sample: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Proxy Anchor over ``atoms``, with cells as the data and atoms as the proxies.

    Parameters
    ----------
    z
        (B, D) latent. Batch-mean centred here when ``center`` (matching
        ``_atom_anchor_loss``'s residual convention) -- a shared offset in z or in the atoms
        pins any cosine objective at its chance floor, which has bitten this model before.
    atoms
        (L, D) LIVE atoms only. The table spans the vocabulary so zero-shot labels can index
        it, but rows past ``n_labels`` are never trained and must not become negatives:
        including them adds thousands of random directions to the repulsion and dilutes every
        metric computed here.
    labels
        (B,) class ids, assumed already filtered to ``[0, L)`` by ``valid``.
    alpha, delta
        Scaling and margin. Larger alpha => sharper hinge; larger delta => the loss keeps
        pushing further past separation (delta=0 turns off as soon as cosines cross).
    neg_only
        Drop the positive term. Use when the anchor already supplies the pull and the only
        thing missing is repulsion.
    z_grad
        Fraction of the gradient that reaches the ENCODER (mirrors ``atom_anchor_z_grad``).
        Exact at 0 (atoms chase z, z untouched) and at 1.
    relax_idx, relax_sim, relax_beta
        PER-PAIR MARGIN RELAXATION from a prior. ``relax_idx`` is (L, k) neighbour indices per
        label and ``relax_sim`` the matching similarities in [0, 1]; the negative margin for a
        cell of class p against proxy q becomes ``delta * (1 - beta * sim_pq)`` for the k
        neighbours of p and plain ``delta`` everywhere else. beta=0 (default) is bit-identical
        to no relaxation.

        Only the NEGATIVE term is touched, and only downward. There is no attractive term here,
        so a prior can decline to push but can never pull -- which is what makes this safe to
        try after every attractive use of a prior in this model failed, several by the prior
        going endogenous. Similarities must be non-negative: a negative one would RAISE the
        margin and push that pair harder than default, a different intervention entirely.

        Motivated by measurement, not symmetry: the hardest negatives are 6.4x enriched for
        genuine co-response (+0.0936 vs a measured null of +0.0146, z = +21.6), so the term is
        spending its force on pairs that share real biology. The absolute similarity is small
        (0.094), so expect a small effect and set beta accordingly.

        WHAT RELAXATION ACTUALLY DOES -- it does NOT reduce the repulsion. The negative term's
        gradient w.r.t. a logit is exp(u_i) / (1 + sum_j exp(u_j)), a softmax over negatives
        that sums to ~1 whenever the sum exceeds 1, which it always does at L=200 in D=128.
        Lowering some logits therefore REDISTRIBUTES the push onto the remaining negatives
        rather than removing it (measured: total atom-gradient conserved to within a few
        percent while the relaxed pair's own share falls). The prior chooses WHERE the
        repulsion budget is spent; it does not change the size of the budget. That is the
        useful reading: separation pressure moves off co-responding pairs and onto unrelated
        ones, which is the allocation you want and is not the same claim as "push less".
    valid
        (B,) bool row mask -- drops synthetic/augmented cells, which must not define proxy
        geometry, and any row whose label is out of range.
    center_atoms
        Subtract the mean over LIVE atoms before the cosine. Defaults True, and the asymmetry it
        fixes is not cosmetic.

        The cells are already centred -- here by the batch mean, and upstream by
        `_center_z_shared`'s per-dataset EMA under invariant_align -- while the atom path
        (`_class_reps`) applies no centring at all. That asymmetry drives a POSITIVE FEEDBACK
        LOOP through the negative term. The negative softmax runs over CELLS for a fixed atom,
        so if the atoms share a component mu_A then cos(z_i, A_p) carries a term <z_i, mu_A>
        that is identical across atoms but varies by cell. Cells aligned with mu_A therefore
        take high weight against EVERY atom, every atom is pushed away from the same cells in
        the same direction, and mu_A grows.

        Measured consequence with center_atoms effectively False: pa/coh_mean RISES over
        training (0.283 -> 0.336 at w=0.25, 0.474 -> 0.571 at w=0.68 neg-only) against a
        random-direction null of 0.071 -- i.e. the repulsion term made the dictionary MORE
        coherent, the exact opposite of its purpose. See project_cosine_shared_offset_collapse:
        a constant component in z OR in the proxies pins a cosine objective at its chance floor,
        and the instruction there is to measure shared_offset_frac FIRST. That is now emitted as
        pa/atom_offset_frac.
    coh_sample
        Atom subsample size for the mutual-coherence diagnostic. The full Gram is L^2, which
        is 80M entries at full scale; a subsample estimates the same quantity for free.
    """
    device = z.device
    L = int(atoms.shape[0])
    y = labels.view(-1).long()
    if valid is None:
        valid = torch.ones_like(y, dtype=torch.bool)
    valid = valid.view(-1).bool() & (y >= 0) & (y < L)
    if int(valid.sum()) < 2 or L < 2:
        return torch.zeros((), device=device), {}

    # PER-CELL SUPERVISION WEIGHT. Both PA terms are log(1 + sum_i exp(u_i)) over cells, so a
    # per-cell weight w_i enters as log(1 + sum_i w_i*exp(u_i)) = log(1 + sum_i exp(u_i + log w_i)).
    # Adding log(w) to the exponent is the numerically stable spelling -- it reuses the existing
    # masked logsumexp instead of introducing a product that can underflow. w=1 is EXACTLY the
    # unweighted loss (log 1 = 0), so passing None or ones is bit-identical to the old behaviour.
    # Weights arrive already rank-normalised and FLOORED (see _efficiency_rank_weight): the floor
    # matters because efficiency==0 is dataset-confounded, so a true zero would delete 27.7% of
    # K562_gwps and 0.0% of Replogle -- a weighting, never an off switch.
    logw = None
    if weights is not None:
        _w = weights.reshape(-1)
        if _w.numel() != y.numel():
            raise ValueError(f"proxy_anchor_loss: weights has {_w.numel()} elements but there "
                             f"are {y.numel()} cells; pass a per-cell weight or None.")
        # THE FLOOR MUST BE FAR BELOW exp(-alpha). A naive clamp_min(1e-6) gives log w >= -13.8,
        # but with alpha=32 the exponents here reach ~+-35, so a 1e-6 weight still contributes
        # exp(35-13.8) -- i.e. NO weight could ever actually remove a cell, and w->0 did not
        # converge to the cell-removed loss (measured gap 4.3e-4 with centering off, where the two
        # must be identical). Clamp at float32 tiny (log = -87.3) and send exact zeros to a
        # finite -1e30, which exp()es to 0 without the inf-minus-inf NaN that -inf would risk.
        _wf = _w.float()
        logw = torch.where(
            _wf > 0,
            torch.log(_wf.clamp_min(torch.finfo(torch.float32).tiny)),
            torch.full_like(_wf, -1e30),
        ).unsqueeze(1)                                                # (B, 1), broadcasts over L

    g = float(z_grad)
    zv = z * g + z.detach() * (1.0 - g)          # exact at g=0 and g=1
    # PER-CELL GRADIENT SCALING, same straight-through trick as z_grad above: the FORWARD value is
    # unchanged (w*zv + (1-w)*zv.detach() == zv) while the backward pass to that cell is scaled by w.
    #
    # WHY THIS IS NEEDED ON TOP OF log(w) IN THE EXPONENT. The exponent form weights a cell's SHARE
    # of its column, d/du_i = w_i*exp(u_i) / (1 + sum_k w_k*exp(u_k)). That works when cells compete
    # (the negative term has ~B-1 per column) but SATURATES to ~1 regardless of w when a cell is the
    # only member of its column -- and at batch 2048 over 8,964 classes most classes contribute
    # exactly one cell, so the POSITIVE term is nearly always in that regime. Measured: at w=0.1 the
    # exponent form alone changed the cell's gradient by 0.1% (ratio 0.999).
    if weights is not None:
        _wg = _w.float().clamp(0.0, 1.0).unsqueeze(1)          # (B, 1)
        zv = zv * _wg + zv.detach() * (1.0 - _wg)
    if center:
        zv = zv - zv[valid].mean(0, keepdim=True)

    # FP32 AND AUTOCAST OFF. alpha=32 turns a cosine into a logit of magnitude ~32; bf16 has
    # 8 mantissa bits, so neighbouring cosines 0.30 and 0.3039 collapse to the SAME logit and
    # the hinge stops being able to order them. Smokes run precision=32, so this class of bug
    # does not show up in one -- it only appears in the bf16 arms that actually train.
    with torch.autocast(device_type=device.type, enabled=False):
        zc = F.normalize(zv.float(), dim=-1)
        av = atoms.float()
        if center_atoms:
            av = av - av.mean(0, keepdim=True)
        ac = F.normalize(av, dim=-1)
        s = zc @ ac.T                                             # (B, L) cosine

        pos = F.one_hot(y.clamp(0, L - 1), L).bool() & valid.unsqueeze(1)
        neg = (~F.one_hot(y.clamp(0, L - 1), L).bool()) & valid.unsqueeze(1)

        # PER-PAIR NEGATIVE MARGIN. Scalar delta unless a prior is supplied, in which case the
        # k prior-neighbours of each cell's own class get a reduced margin and everything else
        # keeps delta exactly. Built as (B, L) -- the same shape as `s`, so no extra asymptotic
        # cost -- rather than an (L, L) dense matrix, which is 321 MB at full scale.
        # SIGN. `relax_sharpen=False` REDUCES the margin toward prior neighbours (the original
        # behaviour). That direction was measured twice and lost: the response-graph arm was
        # -0.0069 top10 SIG vs H4 and its diagnosis was "removes exactly the inter-class
        # competition the readout needs", and text+depmap over the same channel was a null.
        #
        # `relax_sharpen=True` INCREASES it instead, so prior-similar classes are pushed apart
        # HARDER. Two independent measurements point this way: the pairs that actually steal a
        # class's own queries are prior-SIMILAR (AUC ~0.65 as a similarity predictor, vs an
        # atom-space control at 0.73 and every prior at 0.33 in the dissimilarity direction), and
        # the relax arm's own failure says related pairs need MORE competition, not less. It is
        # also what the prototype/hard-negative literature prescribes -- repulsion scaled UP with
        # prototype-prototype similarity (CSPCL; semantically-conditioned negatives) -- as opposed
        # to the false-negative literature, which applies when the "similar" items are the SAME
        # class unlabelled. Ours are distinct labelled perturbations, so the former regime holds.
        #
        # BOUNDED BY CONSTRUCTION: sim is clamped to [0,1], so the margin lands in
        # [delta, (1+beta)*delta]. The literature is consistent that unbounded hard-negative
        # mining destabilises, so beta must stay O(1).
        d_neg = delta
        n_relaxed = 0.0
        if relax_beta > 0.0 and relax_idx is not None and relax_sim is not None:
            d_neg = torch.full_like(s, float(delta))
            nb = relax_idx[y.clamp(0, L - 1)]                      # (B, k) neighbours of own class
            sm = relax_sim[y.clamp(0, L - 1)].to(s.dtype).clamp(0.0, 1.0)
            _sgn = 1.0 if relax_sharpen else -1.0
            d_neg.scatter_(1, nb, float(delta) * (1.0 + _sgn * float(relax_beta) * sm))
            # Never touch a cell's OWN column: it is the positive, not a negative, and the
            # negative mask already excludes it -- but a self-entry in relax_idx would still
            # write into d_neg and make the metric below misleading.
            # The metric counts MODIFIED pairs in whichever direction is active, so it stays
            # comparable across the two signs (it is pinned at k/L either way -- see
            # [[project_pa_priors_null_relaxed_frac_cap]]).
            n_relaxed = float(((d_neg > delta) if relax_sharpen else (d_neg < delta)
                               ).float().sum() / max(float(s.numel()), 1.0))

        # Negative term over ALL proxies -- this is the half that makes the update
        # batch-composition-independent, so it is normalised by |P| = L, not by the number of
        # classes present.
        _u_neg = alpha * (s + d_neg)
        if logw is not None:
            _u_neg = _u_neg + logw
        neg_col = _lse_with_zero(_u_neg, neg)                      # (L,)
        loss_neg = neg_col.mean()

        if neg_only:
            loss = loss_neg
            loss_pos = torch.zeros((), device=device)
        else:
            has_pos = pos.any(0)                                   # (L,)
            _u_pos = -alpha * (s - delta)
            if logw is not None:
                _u_pos = _u_pos + logw
            pos_col = _lse_with_zero(_u_pos, pos)                  # (L,)
            # 1/|P+|: only proxies that actually have a positive this batch. A column with no
            # positive contributes log(1) = 0 and averaging it in would silently scale the
            # positive term by (classes present / L) -- at 200 classes and ~150 present per
            # batch that is a 25% weight error that moves with batch size.
            loss_pos = pos_col[has_pos].sum() / has_pos.sum().clamp_min(1)
            loss = loss_pos + loss_neg

        with torch.no_grad():
            sd = s.detach()
            pf, nf = pos.float(), neg.float()
            n_pos = pf.sum().clamp_min(1.0)
            n_neg = nf.sum().clamp_min(1.0)
            pos_mean = (sd * pf).sum() / n_pos
            neg_mean = (sd * nf).sum() / n_neg
            # THE SATURATION DIAGNOSTIC, and the reason to prefer this over CE. These are the
            # fraction of pairs still inside the margin, i.e. still generating gradient. Under
            # CE they would stay pinned near 1.0 forever; here they should DECAY. If they do
            # not fall over training, the term is behaving like a CE and the margin is set
            # too wide.
            viol_pos = ((sd < delta) & pos).float().sum() / n_pos
            viol_neg = ((sd > -delta) & neg).float().sum() / n_neg
            # Mutual coherence of the dictionary: the classical OMP recovery condition is
            # mu < 1/(2k-1), so at k=3 anything above ~0.2 is trouble for the readout. Watch
            # the MAX, not the mean -- OMP fails on the single worst pair, not the average.
            if L > coh_sample:
                sel = torch.randperm(L, device=device)[:coh_sample]
                asub = ac[sel]
            else:
                asub = ac
            gram = (asub @ asub.T).abs()
            gram.fill_diagonal_(0.0)
            # SHARED-OFFSET FRACTION on the RAW atoms: how much of their energy is one common
            # direction. Measured on `atoms`, never on the centred copy -- centring makes it
            # ~0 by construction, which would report success rather than measure it. At 1/L
            # this is what independent directions give by chance.
            _araw = atoms.float()
            _off = (_araw.mean(0).pow(2).sum()
                    / _araw.pow(2).sum(-1).mean().clamp_min(1e-12))
            metrics = {
                'pa/loss': loss.detach(),
                'pa/pos': loss_pos.detach(),
                'pa/neg': loss_neg.detach(),
                'pa/margin': pos_mean - neg_mean,
                'pa/pos_cos': pos_mean,
                'pa/neg_cos': neg_mean,
                'pa/viol_pos': viol_pos,
                'pa/viol_neg': viol_neg,
                'pa/atom_offset_frac': _off,
                'pa/coh_mean': gram.mean(),
                'pa/coh_max': gram.max(),
                'pa/relaxed_frac': torch.tensor(n_relaxed, device=device),
            }
    return loss.to(z.dtype), metrics
