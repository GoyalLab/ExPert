# SPDX-License-Identifier: MIT
# ExPert: Copyright (c) 2024-2026 Leon Schwartz, Northwestern University, Goyal Lab.
"""f(prior) -> atom: position WEAK-RESPONSE atoms from priors DURING TRAINING.

WHY. PA pulls cells toward their class atom. When a class has a weak knockdown its own cells carry
little signal, so its atom is placed by noise -- and the encoder then spends the whole run being
dragged toward a target that is wrong. Priors cannot beat data where data exists, but where the
measured response is weak they are the better information.

MEASURED OFFLINE (H8 atoms, fold_4 ood, 10,061 queries, paired bootstrap over dumped ranks):
  blending a fused prior into WEAK atoms lifts THOSE classes by +0.125 top10 (K562_essential) /
  +0.037 (gwps), significant in every efficiency band, while the classes we do NOT touch lose a
  comparable amount. The moved atoms become hubs -- mean false-top10 count 9.5 -> 22.3 at the
  operating point. d_GATED is positive in EVERY band, so the prior is not destroying valid
  structure; it is creating broad distractors.
That distinction is the entire reason this belongs in TRAINING and not in predict(): PA's negative
term actively pushes an atom AWAY from cells that are not its own, so hubness is self-limiting
here in a way it can never be on a frozen table.

LOAD-BEARING DESIGN POINTS
  * The target is DETACHED. A prior term with gradient is the struct loss, which cost the readout
    every time: removing struct took response-supported atom edges 0.081 -> 0.319, text edges
    0.442 -> 0.060, and gave the best readout on record.
  * NO direct prior -> encoder path. The encoder improves only because PA's target improved. A
    term that touched z would be priors shaping the latent, which is the thing being avoided.
  * f() is fit on STRONG-response atoms ONLY. Fitting on weak atoms regresses onto exactly the
    noise this exists to replace.
  * RIDGE ONLY, alpha_rel FIXED at 0.5. The map used to be CV-selected per bank between ridge and
    an InfoNCE MLP. Measured (text bank, 6,957 fit atoms, 1,740 held out): EVERY ridge setting
    beats every MLP on both uses -- zero-shot 8.50x vs 6.70x -- and the MLP overfits (in-sample
    cosine 0.40 -> 0.80 with capacity while held-out FALLS 0.065 -> 0.033). The selector also
    could not see what mattered: ridge and MLP differ 2% on its own criterion (held-out top10
    among fit atoms) and 27% on zero-shot.
    alpha_rel 0.5 is the JOINT optimum across the two uses -- 91% of the best held-out top10
    (peak 0.0391 at 0.2) and 98% of the best zero-shot (peak 8.50x at 1.0).
    NOTE this was measured on the TEXT bank, which is the one that carries zero-shot (8.30x vs
    DepMap's 5.90x). DepMap's held-out top10 did prefer an MLP (0.0953 vs 0.0781); using ridge for
    it too is a deliberate simplification, not a measurement.
  * Direction only, fixed scale. The prediction is unit-normalised and rescaled to the STRONG
    atoms' median norm -- exogenous to the atom being pulled. Atom norm sets the OMP coefficients,
    so a term that could inflate or collapse it degrades the readout while every cosine metric
    still looks healthy.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def _unit(M, eps=1e-8):
    return M / M.norm(dim=-1, keepdim=True).clamp_min(eps)


def _ridge(X, Y, alpha_rel):
    """Closed form, so a refit costs one solve and can run every few epochs.

    ALPHA IS RELATIVE to the mean feature variance, never absolute. An absolute lambda tuned on one
    bank silently returns W ~ 0 on another whose features have a different scale -- the same trap
    the OMP lambda hit (see reference_lasso_lam_must_be_relative). Here the banks differ by more
    than scale: the primary text bank arrives raw while the extras arrive centred and normalised.
    """
    d = X.shape[1]
    A = X.T @ X
    lam = float(alpha_rel) * float(torch.diagonal(A).mean().clamp_min(1e-12))
    A = A + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


class PriorAtomPredictor:
    """Fit f(prior) -> atom on strong-response classes, predict the weak ones.

    ``banks`` is [(name, E, valid)] with E row-aligned to the CLASS INDEX -- i.e. exactly
    ``cls_emb._atom_priors`` resolved, which already arrives centred on covered rows and scaled to
    unit mean row-norm. Every (bank, map) pair is scored by held-out retrieval among the fit atoms;
    the winning map per bank is kept and the surviving banks are fused.
    """

    def __init__(self, lam=0.5, ridge_alpha=0.5, min_fit=300, seed=0,
                 cos_floor=0.02, map='ridge', missing_map=None, graph_k=25,
                 graph_loo=False, graph_evidence=None, graph_ev_pow=0.0, graph_cache=None,
                 missing_sim_pow=4.0, **_ignored):
        self.lam = float(lam)
        self.ridge_alpha = float(ridge_alpha)
        self.min_fit, self.seed = int(min_fit), int(seed)
        # MAP FAMILY, separately settable for the two consumers. Measured over 762 (bank x map)
        # configs under the readouts we actually optimise: soft-alpha OOD is FLAT across families
        # (+62 shipped ridge vs +69 best), but zero-shot placement of the MISSING rows is not --
        # every one of the top-20 ZS configs is a graph map and ridge appears zero times
        # (graph25 ~12.0x vs ridge 10.2x). The two paths are independent, so they get independent
        # maps. 'graph' also needs no fitting at all: the prior kNN graph is static, so the target
        # is a weighted mean of CURRENT atom positions and there is nothing to refit.
        self.map = str(map)
        self.missing_map = str(missing_map) if missing_map else None
        self.graph_k = int(graph_k)
        # SHARPER NEIGHBOUR WEIGHTS FOR THE UNTRAINED (zero-shot) ROWS ONLY: graph weight = cos^p
        # instead of cos. The prior cosines inside a gene's top-k are so similar that p=1 is a near-
        # uniform average (24.5 effective neighbours of 25), which pulls fills to the centre (hubs).
        # p=4 at k=25 is the DEFAULT, measured on frozen fold models (12 folds, vs the p=1 refit):
        # zero-shot top-10 +8.8% ID (12/12) / +15.9% OOD (10/12), trained-query top-10 -0.4% with
        # zero-shot rows on; p=8 / softmax / k 50-100 / support- or recall-weighted neighbours were
        # all worse (analysis/readout/fill_fix/). The pull targets of TRAINED rows keep p=1.
        # NOTE this default governs a fill that is FITTED: a checkpoint already carrying
        # `prior_missing_T` keeps whatever fill was written into it, so changing this alone does not
        # change predict() on an existing model -- the fill has to be rebuilt (add_fill.py).
        self.missing_sim_pow = float(missing_sim_pow)
        # LEAVE-ONE-OUT for the graph map. Cross-fitting exists to stop a FITTED map from being
        # trained on the atoms it moves; a neighbour mean has no parameters, so the only leak it
        # can have is an atom appearing among its own neighbours. Excluding exactly that keeps the
        # invariant while every atom sees ALL other atoms as candidates (5 folds hid 20% of them,
        # and f(missing) was built from fold 0's fit set only).
        self.graph_loo = bool(graph_loo)
        # NEIGHBOUR TRUST. `graph_evidence` is a per-ROW evidence rank in [0, 1] (1 = most data,
        # the same orientation as the soft-alpha rank). With graph_ev_pow > 0 a neighbour's weight
        # is sim * evidence^pow, so a rare gene is pulled toward its WELL-DETERMINED prior
        # neighbours rather than toward other rare genes' noise -- the mutual-pull cluster that a
        # plain neighbour mean forms inside a low-evidence module. The TOPOLOGY stays prior-only
        # (top-k by similarity); evidence only reweights inside it, which keeps the cache valid.
        self.graph_evidence = graph_evidence
        self.graph_ev_pow = float(graph_ev_pow)
        # STATIC K-NN CACHE, owned by the caller (a fresh predictor is built every refit). The
        # prior banks never change, so top-k indices and similarities are a function of
        # (bank, fit rows, query rows, k, loo) only; the atoms are looked up fresh each time.
        self.graph_cache = graph_cache
        self.graph_cache_hits = 0
        self.chosen: dict[str, str] = {}      # per-bank map actually used
        # DECLARED IN __init__, not only in fit(). The cross-fit path builds an outer predictor
        # whose fit() is never called -- it merges per-fold results into it -- so any attribute
        # created only inside fit() raises AttributeError in the metrics block downstream.
        self.scores: dict[str, float] = {}
        self.missing_rows: torch.Tensor | None = None
        self.target: torch.Tensor | None = None      # (n_rows, d), rows at `scale`
        self.missing: torch.Tensor | None = None     # (n_missing, d) f() for UNTRAINED rows
        self.mask: torch.Tensor | None = None        # (n_rows,) bool, True where predicted
        self.scale: float = 1.0                      # strong-atom median norm; normalises the loss
        self.cos_floor = float(cos_floor)
        self.coh: dict[str, float] = {}              # hubness watch; see fit()

    def _graph_knn(self, Xtr, Xte, fit_rows, hold_rows, key):
        """top-k (similarity, index-into-fit-rows) per query row; cached when a key is given."""
        cache = self.graph_cache if key is not None else None
        loo = self.graph_loo and fit_rows is not None and hold_rows is not None
        if cache is not None and key in cache:
            fr, hr, kk, lo, v, idx = cache[key]
            if (kk == int(self.graph_k) and lo == loo and fr.shape == fit_rows.shape
                    and hr.shape == hold_rows.shape and torch.equal(fr, fit_rows.cpu())
                    and torch.equal(hr, hold_rows.cpu())):
                self.graph_cache_hits += 1
                return v.to(Xte.device), idx.to(Xte.device)
        Xf, Xh = _unit(Xtr), _unit(Xte)
        n_fit = Xf.shape[0]
        k = max(1, min(int(self.graph_k), n_fit - 1 if n_fit > 1 else 1))
        if loo:
            # position of each query row inside the fit rows (-1 = not a fit row)
            big = int(max(int(fit_rows.max()), int(hold_rows.max()))) + 1
            pos = torch.full((big,), -1, dtype=torch.long, device=Xf.device)
            pos[fit_rows.to(Xf.device)] = torch.arange(n_fit, device=Xf.device)
            self_col = pos[hold_rows.to(Xf.device)]
        vs, ids = [], []
        # CHUNKED over query rows: leave-one-out makes query = fit = every covered atom, so the
        # full similarity matrix is n_atoms^2 (80M entries per bank at full scale).
        for a in range(0, Xh.shape[0], 4096):
            S = Xh[a:a + 4096] @ Xf.T
            if loo:
                sc = self_col[a:a + 4096]
                hit = torch.nonzero(sc >= 0, as_tuple=False).flatten()
                if len(hit):
                    S[hit, sc[hit]] = float('-inf')
            v_, i_ = torch.topk(S, k, dim=1)
            vs.append(v_); ids.append(i_)
        v, idx = torch.cat(vs), torch.cat(ids)
        if cache is not None:
            cache[key] = (fit_rows.cpu(), hold_rows.cpu(), int(self.graph_k), loo,
                          v.detach().cpu(), idx.detach().cpu())
        return v, idx

    def _apply(self, kind, Xtr, Ytr, Xte, fit_rows=None, hold_rows=None, key=None, sim_pow=1.0):
        if kind == 'graph':
            # PRIOR SUPPLIES TOPOLOGY ONLY; the position comes from REAL atoms. Mirrors the
            # offline implementation in analysis/readout/mapcache_build.py exactly: cosine in
            # prior space -> top-k -> clamp negatives -> normalise -> weighted mean of UNIT fit
            # atoms. Weighting raw (non-unit) atoms would let a large-norm neighbour dominate a
            # direction-only target.
            v, idx = self._graph_knn(Xtr, Xte, fit_rows, hold_rows, key)
            k = idx.shape[1]
            w = v.clamp_min(0.0)                      # -inf (the excluded self) -> 0
            if sim_pow != 1.0:
                w = w.pow(sim_pow)
            if (self.graph_evidence is not None and self.graph_ev_pow > 0.0
                    and fit_rows is not None):
                ev = self.graph_evidence.to(w.device, w.dtype)[fit_rows.to(w.device)]
                ev = ev.clamp(0.0, 1.0).pow(self.graph_ev_pow)[idx]
                we = w * ev
                # a query whose neighbours ALL carry ~no evidence keeps the plain similarity
                # weights rather than dividing by ~0 into an arbitrary direction
                ok = we.sum(1, keepdim=True) > 1e-12
                w = torch.where(ok, we, w)
            sw = w.sum(1, keepdim=True)
            w = torch.where(sw > 1e-12, w / sw.clamp_min(1e-12),
                            torch.full_like(w, 1.0 / float(k)))
            return torch.einsum('bk,bkd->bd', w.to(Ytr.dtype), _unit(Ytr)[idx])
        xm, ym = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
        W = _ridge(Xtr - xm, Ytr - ym, self.ridge_alpha)
        return (Xte - xm) @ W + ym

    @torch.no_grad()
    def fit(self, atoms, banks, strong, weak, missing=None, n_trained=None):
        """atoms (n_rows,d) detached; banks [(name,E,valid)]; strong/weak/missing LongTensor rows.

        `missing` are UNTRAINED vocabulary rows. They are predicted with the same map so they can
        be used as EXTRA PA NEGATIVES: filling them makes them PLAUSIBLE, and a plausible wrong
        atom is a real distractor -- measured offline, adding f()-filled rows to the dictionary
        cost -0.0401 top10 on the TRAINED readout while adding the same number of RANDOM-INIT rows
        cost exactly 0.0000. Random rows are never competitive; f() rows are. So a model that
        ships f(missing) must be taught to reject them.
        """
        self.target = self.mask = None
        self.chosen, self.scores = {}, {}
        if strong.numel() < self.min_fit or weak.numel() == 0:
            # WARNING, not info: a skipped fit makes the whole term silently zero, which is the
            # failure mode this repo keeps shipping. State code 5 carries it to the metrics too.
            log.warning('[prior-atoms] FIT SKIPPED -> term is 0: %d strong (need min_fit=%d), '
                        '%d weak', int(strong.numel()), self.min_fit, int(weak.numel()))
            return False

        n_rows, dim = atoms.shape
        acc = torch.zeros(n_rows, dim, device=atoms.device, dtype=atoms.dtype)
        wsum = torch.zeros(n_rows, device=atoms.device, dtype=atoms.dtype)
        gen = torch.Generator(device='cpu').manual_seed(self.seed)

        for i, (name, E, valid) in enumerate(banks):
            if E is None:
                continue
            E = E.to(atoms.device, atoms.dtype)
            cov = (torch.ones(n_rows, dtype=torch.bool, device=atoms.device) if valid is None
                   else valid.to(atoms.device).bool())
            # SAME NORMALISATION FOR EVERY BANK. _expert.py centres and unit-scales the EXTRA banks
            # but the primary (`_atom_text`) arrives raw, so without this the two halves of the
            # fusion reach the fit on different scales and one relative ridge alpha cannot mean the
            # same thing on both.
            if bool(cov.any()):
                E = E - E[cov].mean(0, keepdim=True)
                E = E * cov.unsqueeze(1).to(E.dtype)
                E = E / E[cov].norm(dim=1).mean().clamp_min(1e-6)
            s = strong[cov[strong]]
            w = weak[cov[weak]]
            if s.numel() < self.min_fit or w.numel() == 0:
                log.info('[prior-atoms] %s: skipped (%d strong / %d weak covered)',
                         name, int(s.numel()), int(w.numel()))
                continue

            self.chosen[name] = self.map
            if self.map == 'graph':
                log.info('[prior-atoms] %s: graph k=%d (%d fit, %d pulled)',
                         name, self.graph_k, int(s.numel()), int(w.numel()))
            else:
                log.info('[prior-atoms] %s: ridge alpha_rel=%.2g (%d fit, %d pulled)',
                         name, self.ridge_alpha, int(s.numel()), int(w.numel()))

            P = _unit(self._apply(self.map, E[s], atoms[s], E[w], fit_rows=s, hold_rows=w,
                                  key=(name, 'pull')))
            # Two banks -> (1-lam, lam) so lam keeps its offline meaning; more -> equal weight,
            # where a single lambda has none.
            bw = ((1.0 - self.lam) if i == 0 else self.lam) if len(banks) == 2 else 1.0
            acc[w] += bw * P
            wsum[w] += bw

        # same map, applied to the untrained rows -- predicted here so the caller never refits
        if missing is not None and len(missing):
            acc_m = torch.zeros(len(missing), dim, device=atoms.device, dtype=atoms.dtype)
            wsum_m = torch.zeros(len(missing), device=atoms.device, dtype=atoms.dtype)
            for i, (name, E, valid) in enumerate(banks):
                if E is None or name not in self.chosen:
                    continue
                E = E.to(atoms.device, atoms.dtype)
                cov = (torch.ones(n_rows, dtype=torch.bool, device=atoms.device) if valid is None
                       else valid.to(atoms.device).bool())
                if bool(cov.any()):
                    E = E - E[cov].mean(0, keepdim=True)
                    E = E * cov.unsqueeze(1).to(E.dtype)
                    E = E / E[cov].norm(dim=1).mean().clamp_min(1e-6)
                s_ = strong[cov[strong]]
                mm = missing[cov[missing]]
                if not len(s_) or not len(mm):
                    continue
                P = _unit(self._apply(self.missing_map or self.chosen[name],
                                      E[s_], atoms[s_], E[mm], fit_rows=s_, hold_rows=mm,
                                      key=(name, 'missing'), sim_pow=self.missing_sim_pow))
                bw = ((1.0 - self.lam) if i == 0 else self.lam) if len(banks) == 2 else 1.0
                pos = torch.searchsorted(missing, mm)
                acc_m[pos] += bw * P
                wsum_m[pos] += bw
            mm_ok = wsum_m > 0
            self.missing = torch.zeros_like(acc_m)
            if bool(mm_ok.any()):
                self.missing[mm_ok] = _unit(acc_m[mm_ok]) * atoms[strong].norm(dim=1).median()
            self.missing_rows = missing[mm_ok]
            self.missing = self.missing[mm_ok]
            log.info('[prior-atoms] also predicted %d UNTRAINED rows for use as PA negatives '
                     '(map=%s)', int(mm_ok.sum()), self.missing_map or self.map)

        m = wsum > 0
        if not bool(m.any()):
            log.warning('[prior-atoms] no bank produced predictions -- term is INERT this refit')
            return False
        # Rescale to the STRONG atoms' median norm: exogenous to the atom being pulled, so the
        # term cannot inflate or collapse the magnitude that sets the OMP coefficients.
        scale = atoms[strong].norm(dim=1).median()
        self.target = torch.zeros_like(acc)
        self.target[m] = _unit(acc[m]) * scale
        self.mask = m
        # The loss divides by scale^2, so `prior_atom_weight` means a fixed share of the objective
        # instead of tracking a drifting atom norm. Atom norm was 0.1177 when last measured, so an
        # un-normalised term would sit ~70x smaller than it reads -- the inert-by-negligibility
        # trap, arrived at through the units rather than through the weight.
        self.scale = float(scale.clamp_min(1e-8))

        # PER-ATOM AGREEMENT FLOOR. A global weakness quantile decides WHICH atoms are candidates;
        # this decides whether the prediction for each one is worth acting on. Measured offline, the
        # lowest-agreement ~20% of pulled atoms net ZERO -- at eff/0.075 that decile was 1 helped /
        # 1 hurt -- while still paying the full distraction cost, because a moved atom becomes a
        # false neighbour for OTHER classes' queries. Dropping them at 0.02 kept the gain exactly
        # (+81 -> +81), cut untouched losses (-237 -> -223) and lifted d_GATED 0.0995 -> 0.1233.
        # DIRECTION: the pull works where f() AGREES with the current placement (a refinement) and
        # fails where it disagrees (a relocation) -- strong disagreement means f() is wrong, not the
        # atom. Floors above 0.02 start cutting real gain (+65 at 0.05, +50 at 0.08).
        # UNTRAINED ROWS ARE EXEMPT. The floor asks "does f() agree with where the data put this
        # atom" -- a question that only has meaning when the data put it somewhere. An untrained row
        # is still at randn*0.02, so cos(A,T) ~ 0 against ANY prediction and the floor would discard
        # every one of them. The zero-shot path routes unseen rows through `weak`, so flooring them
        # collapsed ZS from 10.20x to 3.10x chance. Rows >= n_trained keep their prediction whatever
        # the agreement; None means every row is treated as trained.
        if self.cos_floor > -1.0 and bool(m.any()):
            _idx = torch.nonzero(m).flatten()
            if n_trained is not None:
                _idx = _idx[_idx < int(n_trained)]
            _cs = (_unit(atoms[_idx]) * _unit(self.target[_idx])).sum(-1)
            _drop = _idx[_cs < self.cos_floor]
            if len(_drop):
                m[_drop] = False
                self.target[_drop] = 0
                self.mask = m
            log.info('[prior-atoms] cos-floor %.3g: %d/%d trained targets kept (%d below floor); '
                     '%d untrained rows exempt',
                     self.cos_floor, len(_idx) - int(len(_drop)), len(_idx), int(len(_drop)),
                     int(m.sum()) - len(_idx) + int(len(_drop)))
            if not bool(m.any()):
                log.warning('[prior-atoms] cos-floor removed EVERY target -- term is 0 this refit')
                return False

        # HUBNESS WATCH. Offline, the ENTIRE cost of this mechanism was distraction: the moved
        # atoms became hubs (mean false-top10 9.5 -> 22.3) while their own classes improved in
        # every band. PA's negative term should suppress that here -- but "should" is how inert
        # terms ship. Mean off-diagonal cosine is the cheap standing proxy: coh_pred >> coh_strong
        # means the predictions crowd together more than real atoms do, i.e. the hub is forming.
        # Once per REFIT on a subsample, so it costs nothing.
        g2 = torch.Generator(device='cpu').manual_seed(self.seed + 1)

        def _coh(M):
            if M.shape[0] < 32:
                return float('nan')
            if M.shape[0] > 2000:
                M = M[torch.randperm(M.shape[0], generator=g2)[:2000].to(M.device)]
            U = _unit(M)
            k = U.shape[0]
            S = U @ U.T
            return float((S.sum() - S.diagonal().sum()) / (k * (k - 1)))

        self.coh = {'strong': _coh(atoms[strong]),
                    'target': _coh(atoms[torch.nonzero(m, as_tuple=False).flatten()]),
                    'pred': _coh(self.target[m])}
        log.info('[prior-atoms] mean off-diag cosine: strong %.4f / target-atoms %.4f / '
                 'PREDICTIONS %.4f   (pred >> strong = hubs forming)',
                 self.coh['strong'], self.coh['target'], self.coh['pred'])
        log.info('[prior-atoms] predicted %d/%d weak atoms from %d strong; maps=%s lam=%.2f '
                 'target_norm=%.4f', int(m.sum()), int(weak.numel()), int(strong.numel()),
                 self.chosen, self.lam, float(scale))
        return True
