# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Translate configs and checkpoint kwargs written by pre-release ExPert versions.

Every knob removed from the package is listed here with the value(s) that meant "off". A legacy
config or checkpoint may carry such a knob only at an off value: it is then dropped, and the result
behaves exactly as before. A removed knob set to anything else means the source used a mechanism
this package no longer has, and translation raises instead of silently training or loading a
different model.
"""
from __future__ import annotations

import copy
import logging

log = logging.getLogger(__name__)

_ANY = object()      # dropped whatever its value (pure bookkeeping, never changed behaviour)

# section -> {knob: tuple of off values | _ANY}. Sections: 'model' = ExPert/XPert __init__ kwargs,
# 'plan' = train.plan_kwargs, 'schedules' = train.plan_kwargs.anneal_schedules (weight must be off).
_OFF0, _OFFF = (0, 0.0), (False, None)
_OMP = ('k', 'min_gain', 'min_gain_anneal', 'min_gain_learn', 'tau', 'relax', 'gain', 'loss_kind', 'detach_z',
        'center', 'logit_mode', 'res_weight', 'every')
_SRR = ('k', 'T', 'min_z', 'n_shuffle', 'every', 'gate_scope', 'edge_filter', 'skip_primary')
REMOVED: dict[str, dict[str, object]] = {
    'model': {
        # deprecated shims (already ignored before the release)
        'anchor_ds_center': _OFFF, 'lcksvd_weight': _OFF0, 'lcksvd_code_ce_weight': _OFF0,
        'lcksvd_code_l1_weight': _OFF0, 'lcksvd_min_gain_learn': _OFFF,
        # OMP rollout loss
        'omp_rollout_weight': _OFF0, **{f'omp_rollout_{k}': _ANY for k in _OMP},
        # recursive (OR-PIT) decoder
        'use_recursive_decode': _OFFF, 'extra_recursive_decode_kwargs': _ANY,
        # struct-rerank prior loss
        'struct_rerank_weight': _OFF0, **{f'struct_rerank_{k}': _ANY for k in _SRR},
        # mixture loss, comb loss + synthetic generation (no weight knobs at model level: weights are schedules)
        **{f'mix_{k}': _ANY for k in ('k_min', 'k_max', 'per_step', 'alpha', 'space')},
        **{f'comb_{k}': _ANY for k in ('k_min', 'k_max', 'per_step', 'alpha', 'alpha_jitter', 'eta', 'min_cells',
                                        'var_mode', 'target_atoms', 'metric', 'margin', 'n_neg', 'sample',
                                        'real_frac', 'cross_dataset')},
        'syn_source': _ANY,
        # atom-struct prior loss + cross-dataset atom uncertainty U
        'atom_struct_weight': _OFF0, 'atom_xds_u_weight': _OFF0,
        **{f'atom_struct_{k}': _ANY for k in ('nbrs', 'push', 'q', 'mode', 'gate_thr', 'gate_q', 'T', 'T_text',
                                              'rowgate', 'nbr_mode', 'entmax_alpha', 'conf_pow', 'conf_alpha',
                                              'conf_norm', 'conf_floor', 'agree_cap')},
        **{f'atom_xds_u_{k}': _ANY for k in ('T', 'refresh', 'clamp', 'form', 'standardize')},
        # atom coherence / residual
        'atom_coherence_weight': _OFF0, 'atom_residual_weight': _OFF0,
        **{f'atom_coherence_{k}': _ANY for k in ('mode', 'T', 'max_c')}, 'atom_residual_T': _ANY,
        # atom anchor / recon / decode, and the GECO controllers
        'atom_anchor_weight': _OFF0, 'atom_recon_weight': _OFF0, 'atom_decode_weight': _OFF0,
        'atom_anchor_z_grad': _ANY, 'atom_anchor_mode': _ANY, 'atom_anchor_center': _ANY, 'atom_recon_hidden': _ANY,
        'geco_kl_rate': _OFF0, 'geco_anchor_target': _OFF0,
        **{f'geco_{k}': _ANY for k in ('kl_lr', 'kl_rate_start', 'kl_rate_epochs', 'kl_rate_schedule', 'kl_lam_init',
                                       'anchor_lr')},
        'anchor_mu_decay': _ANY, 'anchor_mu_min_n': _ANY,
        # ---- batch 3 (_xpert.py) ----
        # atom warm-up / EMA hand-off
        'atom_warmup_epochs': _OFF0, 'atom_pca_rank': _ANY,
        # prior-atom variants other than the hard gate (bayes/soft/agree/fisher/fwd-blend/margin/cell/nbr)
        'prior_atom_gamma': _ANY, 'prior_atom_graph_min_ev': (-1, -1.0, None), 'prior_atom_anchor_every': _OFF0,
        'prior_cell_weight': _OFF0, 'prior_cell_eff_weight': _ANY,
        'prior_nbr_weight': _OFF0, **{f'prior_nbr_{k}': _ANY for k in ('k', 'neg', 'scale', 'eff_weight')},
        'prior_atom_fwd_blend': _OFFF, 'prior_atom_fwd_alpha_max': _ANY, 'prior_atom_fwd_fill_missing': _ANY,
        'prior_atom_loss_form': ('l2',), **{f'prior_atom_margin_{k}': _ANY for k in ('kappa', 'alpha_max', 'freeze_epoch')},
        **{f'prior_atom_bayes_{k}': _ANY for k in ('lo_q', 'bins', 'momentum', 'min_t', 'match_dose')},
        'prior_atom_pull_k': _ANY, 'prior_atom_gate_refresh': _ANY,
        # pseudobulk recon / bank alignment / input pseudobulk bank
        'bank_align': _OFFF, 'bank_pca_init': _OFFF, 'bank_rel_loss_weight': (False, 0, 0.0, None),
        'bank_align_min_rel': _ANY, 'bank_align_warmup_cov': _ANY, 'pbulk_gene_rel_weight': (False, 0, 0.0, None),
        'pbulk_bank_weight': ('none', None), 'use_pseudobulk_bank': _OFFF, 'extra_pbulk_kwargs': (None, {}),
        # input / decoder generality
        'decoder_ctrl_offset': _OFFF, 'cpm_active': _OFFF, 'impute_missing': _OFFF, 'decode_ctx_emb': _OFFF,
        'link_latents': _OFFF, 'global_scale': _OFFF, 'encode_cont_only': _OFFF,
        'decode_context_projection': _OFFF, 'encoder_type': ('default',), 'mirror_recon_mask': _ANY,
        'n_latent_local': (None,), 'n_latent_global': (None,),
        # augmentation, GroupDRO, split latent / zn adversary, hierarchy
        'use_augmentation': _OFFF, 'extra_aug_kwargs': (None, {}), 'mixup_group_map': (None,),
        'use_group_dro': _OFFF, 'group_dro_eta': _ANY, 'group_dro_floor': _ANY,
        'n_p': _OFF0, 'zn_adv_weight': _OFF0,
        **{k: _ANY for k in ('split_decoder_label', 'split_label_dim', 'zn_adv_grl', 'zn_adv_hidden')},
        'use_hierarchical_labels': _OFFF, 'use_hierarchical_clip': _OFFF,
        'use_learnable_hierarchy_temperatures': _OFFF,
        **{k: _ANY for k in ('hierarchy_inference', 'module_quantile', 'pathway_quantile')},
        # adversarial / invariance family. CAVEAT: loss() used to default dann_weight to 0.1 with use_cdan True, so
        # a legacy config WITHOUT a dann_weight schedule trained with DANN; that cannot be detected from its keys.
        'use_cdan': _ANY, 'cdan_cond_on_head': _OFFF, 'cdan_freeze_label_emb': _ANY, 'cdan_cond_dim': _ANY,
        'cdan_label_dim': _ANY, 'wc_adv_cond_dim': _ANY, 'dann_group_map': (None,), 'extra_cdan_kwargs': (None, {}),
        'shared_k': _OFF0,
        **{f'shared_{k}': _ANY for k in ('rff_dim', 'm', 'ema', 'refresh_every', 'min_cells', 'max_classes')},
        # graph / DEG / GNN
        'class_encoder': ('emb',), 'graph_reg_weight': _OFF0, 'proto_div_weight': _OFF0, 'proxy_agg_tau': _ANY,
        'extra_gnn_kwargs': (None, {}), 'extra_deg_kwargs': (None, {}), 'extra_f_kwargs': (None, {}),
        'deg_recon_normalize': _ANY,
        # head and CLIP-alignment variants (the cosine CE head and plain CLIP logits remain)
        'cls_align_loss': ('clip',), 'cls_head_loss': ('ce',), 'cls_head_hidden': _OFF0, 'extra_cls_kwargs': (None, {}),
        'cls_head_arc_margin': _OFF0, 'head_soft_targets': _OFFF, 'head_soft_T': _ANY, 'head_soft_alpha': _ANY,
        'asl_clip': _ANY, 'asl_gamma_neg': _ANY, 'asl_gamma_pos': _ANY, 'softkl_eps': _ANY, 'bce_pos_weight': _ANY,
        'cls_align_center': _OFFF, 'ep_balanced_anchor': _OFFF, 'ep_drift_metric': _OFFF, 'ep_balance_detach_ce': _ANY,
        'use_efficiency_clip_weight': _OFFF, 'mask_zero_efficiency': _OFFF,
        # bookkeeping / weighting no longer registered
        'track_class_efficiency': _OFFF, 'use_cls_weights': _ANY, 'use_ctx_weights': _OFFF, 'use_ds_sim_weights': _OFFF,
        # ---- batch 3 (_expert.py): model-level knobs ExPert used to pop for the banks / teachers / HCE / ctrl_pc ----
        'pbulk_recon_online': _OFFF, 'pbulk_shift_online': _OFFF, 'teacher_online': _OFFF,
        **{f'pbulk_recon_{k}': _ANY for k in ('granularity', 'min_cells', 'panel', 'refresh_every')},
        **{f'pbulk_shift_{k}': _ANY for k in ('min_cells', 'refresh_every', 'warmup_steps', 'subspace_k',
                                              'subspace_alpha', 'subspace_min_ds')},
        'pbulk_bank_path': (None,), 'pbulk_shift_bank_path': (None,), 'teacher_sim_path': (None,),
        **{f'teacher_{k}': _ANY for k in ('sim_min_rel', 'sim_min_cos', 'panel', 'min_cells', 'min_cos', 'min_rel',
                                          'topk', 'eff_gate', 'refresh_every')},
        'hce': _OFFF, **{f'hce_{k}': _ANY for k in ('levels', 'form', 'alpha', 'level_weights', 'min_rel', 'min_cells',
                                                   'refresh_every', 'min_ready_frac')},
        'ctrl_pool_per_ds': (0, None), 'ctrl_pc_k': (0, None),
        **{f'ctrl_pc_{k}': _ANY for k in ('ref_dim', 'space', 'n_views')},
        'mixup_group_prefixes': (None, []),
        # ExPert.__init__: the per-gene embedding table (never read by the model) and an unused flag
        'use_gene_emb': _ANY, 'experimental': _ANY,
        # ---- batch 4 ----
        # ExPert.__init__ options with a single supported value / never read
        'cls_text_dict': (None,), 'use_cpm': _OFFF, 'use_ds_anchors': _OFFF, 'scale': _OFFF,
        'force_full_cls_emb': _ANY, 'reconstruct_raw_x': (True,),
        # XPert.__init__ knobs whose consumers are gone
        'ctrl_pc_dropout': _ANY, 'ctrl_proj_dim': (0, None), 'ctrl_proj_n': _ANY,
        'use_reconstruction_control': _ANY, 'use_kl_control': _ANY, 'l1_lambda': _ANY, 'l2_lambda': _ANY,
        'l_mask': _ANY,
    },
    'plan': {
        # grad balancer, LR adaptation, auto temperature controller (all default-off, not in the v0.1.0 config)
        'auto_temperature_args': (None, {}), 'grad_balance': (None, False, {}), 'lr_adapt_recon': _OFFF,
        **{f'lr_adapt_{k}': _ANY for k in ('thresh', 'ema', 'floor', 'groups', 'metric', 'warmup_epochs',
                                          'ref_epochs')},
        # module freezing / per-module LR (only reachable through the removed from_base_model fine-tune)
        'freeze_modules': (None, {}), 'module_weight_decay': (None, {}), 'lr_modules': (None, {}),
        'freeze_encoder_epoch': (None,), 'freeze_global_encoder_epoch': (None,), 'freeze_decoder_epoch': (None,),
        'soft_freeze_lr': _ANY, 'gene_emb': (None,),
        # plot / logging-only knobs whose code is gone
        'plot_kl_distribution': _ANY, 'plot_ctx_invariance': _ANY, 'async_logging': _ANY,
        'log_random_predictions': _ANY, 'cache_n_epochs': _ANY,
        'cache_group_aware': (True,),          # the group-aware cache is the only implementation left
    },
    # ExPert.setup_anndata kwargs (config section model_setup): context / feature-mask / gene-embedding / dataset-
    # similarity inputs, PPI edge graphs and the DEG prior -- none used by the released model
    'model_setup': {
        'context_emb_uns_key': (None,), 'feature_mask_varm_key': (None,), 'gene_emb_varm_key': (None,),
        'dataset_sim_uns_key': (None,), 'edge_graph': (None,), 'edge_channel': _ANY, 'edge_min_score': _ANY,
        'ppi_edge_path': (None,), 'ppi_channel': _ANY, 'ppi_min_score': _ANY,
        'deg_efficiencies_dir': (None,), 'deg_min_datasets': _ANY, 'deg_promiscuity_cap': _ANY,
        'deg_recon_lambda': _ANY, 'deg_exclude_datasets': _ANY, 'deg_callability_h5ad': _ANY,
    },
    # data / train config sections
    'data': {'zs_subset_path': (None,), 'use_balanced_epochs': _OFFF, 'target_per_group': _ANY, 'max_ratio': _ANY,
             'min_group_samples': _ANY},
    'train': {'swa_enabled': _OFFF, **{f'swa_{k}': _ANY for k in ('lrs', 'epoch_start', 'annealing_epochs',
                                                                    'annealing_strategy', 'device')}},
    # nested kwarg dicts of the model config: {config key: {knob: off values | _ANY}}
    'nested': {
        'extra_aligner_kwargs': {'use_controller': _OFFF, 'freeze_cls_proj': _ANY},
        # proxy-anchor margin relaxation graph
        'extra_proxy_anchor_kwargs': {'relax_beta': _OFF0, **{f'relax_{k}': _ANY for k in (
            'sharpen', 'k', 'min_ds', 'priors', 'prior_w', 'mode', 'shuffle', 'use_response')},
            **{f'refresh_{k}': _ANY for k in ('every', 'backoff', 'max')}},
        # ClipLoss -> PrototypeScorer: the CLIP-training options; other extra_clip_kwargs keys are ignored
        'extra_clip_kwargs': {'use_soft_targets': _OFFF, 'use_cls_prototypes': _OFFF, 'use_reverse': _OFFF,
                              'arc_margin': _OFF0, 'supervision_dropout_rate': _OFF0, 'ds_sim_weights': (None,),
                              'cls_weights': (None,)},
    },
    # schedule name -> True (its weight must be 0) | _ANY (an auxiliary factor of a removed loss: any value)
    'schedules': {
        'rec_decode_weight': True, 'rec_decode_stop_weight': _ANY, 'rec_decode_skip_first': _ANY,
        'rec_decode_orth_weight': _ANY, 'rec_decode_norm_weight': _ANY,
        'mixture_weight': True, 'comb_weight': True, 'atom_anchor_weight': True, 'atom_struct_weight': True,
        'atom_coherence_weight': True, 'atom_residual_weight': True,
        # batch 3: adversarial / invariance, adapter regularisers, CLIP anchor, distillation, banks, gene anchor
        **{k: True for k in ('anchor_weight', 'adapter_drift_weight', 'adapter_geom_weight', 'adapter_ortho_weight',
                             'ctx_mmd_weight', 'dann_weight', 'wc_adv_weight', 'distill_weight', 'gene_anchor_weight',
                             'pbulk_recon_weight', 'shared_offset_weight', 'contrastive_weight', 'consistency_weight',
                             'ep_balance_weight', 'center_weight', 'deg_geom_weight', 'deg_distill_weight',
                             'cond_align_weight', 'hce_weight')},
        **{k: _ANY for k in ('dann_grl_lambda', 'wc_adv_grl_lambda', 'distill_T', 'distill_holdout_frac')},
        # batch 5: CLIP alignment is no longer a training option (the prototype logits remain)
        'cls_align_weight': True,
    },
}


def _is_off(value, off) -> bool:
    if off is _ANY:
        return True
    return any(value == o or (isinstance(o, float) and isinstance(value, (int, float)) and float(value) == o)
               for o in off)


def _schedule_off(sched) -> bool:
    """An anneal schedule is off when every weight it can reach is 0."""
    if not isinstance(sched, dict):
        return sched in (0, 0.0, None, False)
    return all(float(sched.get(k, 0) or 0) == 0.0 for k in ('weight', 'max_weight', 'min_weight'))


# Flat knob families folded into one dict each: `<prefix><key>` -> `<dict>[<key>]`.
FOLDED: dict[str, str] = {
    'prior_atom_': 'extra_prior_atom_kwargs',
    'proxy_anchor_': 'extra_proxy_anchor_kwargs',
}


def _fold_flat(out: dict, source: str) -> None:
    for pfx, dname in FOLDED.items():
        flat = {k[len(pfx):]: out.pop(k) for k in [k for k in out if k.startswith(pfx)]}
        if not flat:
            continue
        sub = dict(out.get(dname) or {})
        clash = {k: (sub[k], v) for k, v in flat.items() if k in sub and sub[k] != v}
        if clash:
            raise ValueError(f'{source} sets {sorted(clash)} both as flat {pfx}* knobs and in {dname} '
                             f'with different values: {clash}')
        for k, v in flat.items():
            sub.setdefault(k, v)
        out[dname] = sub


def translate_model_kwargs(kwargs: dict, source: str = 'config') -> dict:
    """Drop removed model knobs that sit at an off value (raise on any that do not); fold flat families."""
    out = dict(kwargs)
    dropped, live = [], []
    for k, off in REMOVED['model'].items():
        if k in out:
            (dropped if _is_off(out[k], off) else live).append(k)
    if live:
        raise ValueError(f'{source} sets removed model knob(s) to a non-off value: '
                         f'{ {k: out[k] for k in live} }. This package no longer implements them.')
    for k in dropped:
        out.pop(k)
    _fold_flat(out, source)
    for sect, knobs in REMOVED['nested'].items():
        sub = out.get(sect)
        if not isinstance(sub, dict):
            continue
        sub = dict(sub)
        bad = [k for k, off in knobs.items() if k in sub and not _is_off(sub[k], off)]
        if bad:
            raise ValueError(f'{source} sets removed {sect} knob(s) to a non-off value: { {k: sub[k] for k in bad} }')
        for k in knobs:
            sub.pop(k, None)
        out[sect] = sub
    if dropped:
        log.info('[legacy] %s: dropped %d removed model knob(s) at their off value', source, len(dropped))
    return out


def translate_setup_kwargs(kwargs: dict, source: str = 'config') -> dict:
    """setup_anndata kwargs: drop removed inputs at their off value; raise on any that is set."""
    out = dict(kwargs)
    live = [k for k, off in REMOVED['model_setup'].items() if k in out and not _is_off(out[k], off)]
    if live:
        raise ValueError(f'{source} sets removed setup_anndata input(s): { {k: out[k] for k in live} }. '
                         f'This package no longer implements them.')
    for k in REMOVED['model_setup']:
        out.pop(k, None)
    return out


def translate_config(cfg: dict) -> dict:
    """Full run config (model / train.plan_kwargs / anneal_schedules) -> current schema."""
    cfg = copy.deepcopy(cfg)
    if 'model' in cfg:
        cfg['model'] = translate_model_kwargs(cfg['model'], 'config')
    if isinstance(cfg.get('model_setup'), dict):
        cfg['model_setup'] = translate_setup_kwargs(cfg['model_setup'], 'config')
    for sect in ('data', 'train'):
        sub = cfg.get(sect)
        if isinstance(sub, dict):
            bad = [k for k, off in REMOVED[sect].items() if k in sub and not _is_off(sub[k], off)]
            if bad:
                raise ValueError(f'config {sect}: removed option(s) set: { {k: sub[k] for k in bad} }')
            for k in REMOVED[sect]:
                sub.pop(k, None)
    plan = ((cfg.get('train') or {}).get('plan_kwargs')) or {}
    live = []
    for k, off in REMOVED['plan'].items():
        if k in plan:
            if _is_off(plan[k], off):
                plan.pop(k)
            else:
                live.append(k)
    sched = plan.get('anneal_schedules') or {}
    # Pre-release loss() defaulted cls_align_weight to 1.0: a legacy config WITHOUT that schedule trained CLIP
    # alignment, which this package no longer has. A config is legacy if it carries any removed schedule name.
    if sched and 'cls_align_weight' not in sched and any(k in sched for k in REMOVED['schedules']):
        raise ValueError('legacy config has no cls_align_weight schedule: the pre-release default trained CLIP '
                         'alignment at weight 1.0, which this package no longer implements.')
    for k, rule in REMOVED['schedules'].items():
        if k in sched:
            if rule is _ANY or _schedule_off(sched[k]):
                sched.pop(k)
            else:
                live.append(f'anneal_schedules.{k}')
    if live:
        raise ValueError(f'config switches on removed mechanism(s): {live}')
    return cfg


# State-dict keys (relative to the XPert module, regex full match) of buffers, parameters and submodules that
# belonged to removed mechanisms. A pre-release checkpoint may carry them; they are dropped IN MEMORY before the
# strict load, and ONLY when the current module does not have the key -- any other unexpected key still fails.
# Listing a key early is harmless: while the module still registers it, it is loaded as before.
REMOVED_STATE: tuple[str, ...] = (
    # adversarial / invariance heads (built but unused in v0.1.0)
    r'_dann_head\..+', r'_wc_adv_proj', r'_wc_adv_head\..+', r'_cdan_.+', r'_dann_group_map', r'_zn_adv_head\..+',
    # proxy-anchor relax graph, atom warm-up hand-off, never-read registrations
    r'_pa_nbr_idx', r'_pa_nbr_sim', r'_atom_handoff', r'ctx_centroids', r'ctx_centroid_counts', r'cls_weights',
    # unused per-gene embedding table
    r'gene_embedding',
    # atom subsystems
    r'atom_u', r'anchor_mu', r'anchor_mu_n', r'atom_recon_head\..+', r'omp_min_gain_p', r'recursive_decoder\..+',
    # prior-atom variants, banks, teachers, HCE, DRO, scaling, tracking
    r'prior_fwd_.+', r'prior_margin_delta', r'pbulk_.+', r'_pbo_.+', r'_pb_.+', r'_shift_.+', r'_tch_.+', r'_hce_.+',
    r'_mixup_group_map', r'group_dro_q', r'group_dro_q_align', r'group_dro_q_head', r'running_mean', r'running_std', r'n_batches_seen',
    r'class_eff_sum', r'class_eff_count', r'log_t_(gene|module|pathway)', r'_split_label_emb\..+',
    # graph / DEG / GNN
    r'_greg_e[iw]', r'deg_.+', r'graph_gene_encoder\..+', r'deg_encoder\..+', r'f_text_encoder\..+',
    # feature masks, dataset anchors, context inputs, imputation, non-cosine / MLP heads
    r'ds_mean_anchors', r'ds_std_anchors', r'feat_masks', r'shared_mask', r'ctx_weights', r'ctx_emb\..+',
    r'cls_sim\..+', r'imp_.+', r'classifier\..+', r'_cls_head_mlp\..+',
    # ClipLoss buffers (only allocated by CLIP-training options)
    r'clip_loss\..+',
)
_REMOVED_STATE_RE = None


def drop_removed_state(module, state_dict, prefix: str) -> list[str]:
    """Load pre-hook body: pop keys of removed mechanisms from a legacy `state_dict` (in place, in memory)."""
    import re
    global _REMOVED_STATE_RE
    if _REMOVED_STATE_RE is None:
        _REMOVED_STATE_RE = re.compile('|'.join(f'(?:{p})' for p in REMOVED_STATE))
    own = set(module.state_dict().keys())
    drop = [k for k in list(state_dict) if k.startswith(prefix) and k[len(prefix):] not in own
            and _REMOVED_STATE_RE.fullmatch(k[len(prefix):])]
    for k in drop:
        state_dict.pop(k)
    if drop:
        log.info('[legacy] checkpoint: dropped %d state entr%s of removed mechanisms (%s%s)', len(drop),
                 'y' if len(drop) == 1 else 'ies', ', '.join(sorted({k[len(prefix):].split('.')[0] for k in drop})[:8]),
                 ' ...' if len(drop) > 8 else '')
    return drop
