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

"""Table S4: hyperparameters of the shipped ExPert v0.1.0, ACTIVE settings only. A knob that is set but has no
effect in this configuration (a loss at weight 0, a module that is built but never trained, options of an encoder
or map type that is not used) is left out.

Values are read from the run's hparams.yaml (the resolved config, not the launch config), the checkpoint (layer
shapes) and the training tfevents, never typed. Activity is ASSERTED: every listed loss weight must be the logged
value at every step, every scheduled loss weight left out must be 0 at every step, the free classifier head must
have 0 training steps, and every listed layer width must match the checkpoint. A config that drifts from this
table fails the build instead of producing a stale row.

Writes --out-dir/Table_S4.csv + Table_S4.xlsx.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

BANK_NAME = {'text': 'gene-description text', 'depmap_dstd': 'DepMap', 'string': 'STRING'}
# scheduled loss weights that must be 0 at EVERY step for the rows below to be the whole objective
INERT_SCHEDULED = ['cls_align_weight', 'cls_head_weight', 'anchor_weight', 'dann_weight', 'dann_grl_lambda',
                   'wc_adv_weight', 'wc_adv_grl_lambda', 'ctx_mmd_weight', 'comb_weight', 'pbulk_recon_weight',
                   'rec_decode_weight', 'shared_offset_weight', 'distill_weight', 'gene_anchor_weight',
                   'adapter_drift_weight', 'adapter_geom_weight', 'adapter_ortho_weight', 'gate_eff', 'T_gamma']


def scalars(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(path, size_guidance={'scalars': 0})
    ea.Reload()
    return {t: np.array([e.value for e in ea.Scalars(t)]) for t in ea.Tags()['scalars']}


def const(S, tag, want):
    v = S[tag]
    assert np.allclose(v, want), f'{tag}: logged {v.min():.4g}..{v.max():.4g}, table says {want}'
    return want


def fmt(x):
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int, np.integer)):
        return f'{int(x):,}'
    if isinstance(x, float):
        return f'{x:g}'
    return str(x)


def main(a):
    H = yaml.safe_load(Path(a.hparams).read_text())
    M, D, P, T = H['model_params'], H['data_params'], H['plan_params'], H['train_params']
    E, DEC, CLS = M['extra_encoder_kwargs'], M['extra_decoder_kwargs'], M['extra_cls_emb_kwargs']
    AS = P['anneal_schedules']
    S = scalars(a.events)
    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    sd, ad = ck['model_state_dict'], ck['attr_dict']
    reg = ad['registry_']['field_registries']

    def shp(k):
        if k not in sd:
            raise KeyError(f'{k} not in checkpoint; same module: '
                           f'{[(n, tuple(v.shape)) for n, v in sd.items() if n.startswith(k.split(".")[0] + ".")]}')
        return tuple(sd[k].shape)

    # ---- activity checks ----
    for k in INERT_SCHEDULED:
        assert float(AS[k]['weight']) == 0.0, f'{k} is scheduled non-zero: {AS[k]}'
        assert np.all(S[f'{k}_step'] == 0), f'{k} logged non-zero'
    assert ad['cls_head_training_']['steps'] == 0, 'free classifier head was trained'
    assert float(M['proxy_anchor_relax_beta']) == 0 and np.all(S['pa/relaxed_frac_train'] == 0)
    assert not any(M[k] for k in ('use_cdan', 'use_group_dro', 'use_augmentation', 'use_recursive_decode',
                                  'pbulk_recon_online', 'pbulk_shift_online', 'teacher_online', 'link_latents'))
    assert not E['use_emb'] and not DEC['use_film'] and CLS['adapter_mode'] == 'identity'
    assert np.all(S['pa/eff_w_state_train'] == 4), 'efficiency weight on Proxy Anchor did not fire'
    assert np.all(S['prior_atom/state_train'] == 6), 'prior pull was not live every epoch (or tripped)'
    assert not P['reduce_lr_on_plateau'] and not P['use_cosine_restart'], 'learning rate is scheduled'
    assert M['prior_atom_gate'] == 'eff' and M['prior_atom_map'] == 'graph'   # gamma / ridge_alpha unused

    n_ep = int(S['epoch'].max()) + 1
    assert n_ep == T['max_epochs']
    n_steps = len(S['train_loss_step']) * 10                      # logged every 10 steps (lightning default)
    rl = const(S, 'rl_weight_step', float(AS['rl_weight']['weight']))
    kl = AS['kl_weight']
    assert kl['schedule'] == 'linear' and float(kl['min_weight']) == 0
    kl_last = float(kl['max_weight']) * (n_ep - 1) / int(P['n_epochs_warmup'])
    assert np.isclose(S['kl_weight_epoch'][-1], kl_last), (S['kl_weight_epoch'][-1], kl_last)
    lam_eff = const(S, 'lambda_eff_step', float(AS['lambda_eff']['weight']))
    w_pa = const(S, 'weights/proxy_anchor_train', float(M['proxy_anchor_weight']))
    w_pr = const(S, 'weights/prior_atom_train', float(M['prior_atom_weight']))
    w_ng = const(S, 'weights/prior_neg_train', float(M['prior_atom_neg_weight']))
    assert np.isclose(S['prior_atom/ramp_train'][0], 1 / M['prior_atom_ramp_epochs'])
    n_pulled = int(S['prior_atom/n_train'][-1])
    assert np.allclose(S['prior_atom/pull_frac_train'], M['prior_atom_pull_q'], atol=1 / n_pulled)   # integer count

    # ---- sizes, from the checkpoint ----
    n_genes = reg['X']['state_registry']['n_vars']
    n_ds = reg['batch']['summary_stats']['n_batch']
    n_lab = reg['labels']['summary_stats']['n_labels']
    n_vocab, d_atom = shp('cls_emb.embedding.weight')
    n_hid, n_lat = M['n_hidden'], M['n_latent']
    assert shp('z_encoder.encoder.encoder.layers.0.block.0.weight') == (n_hid, n_genes)
    assert len([k for k in sd if k.startswith('z_encoder.encoder.layers.') and k.endswith('ff.1.weight')]) \
        == E['n_layers']
    assert shp('z_encoder.mean_encoder.weight') == (n_lat, n_hid) and d_atom == M['n_shared'] == n_lat
    d_cat = shp('decoder.cat_embeddings.0.weight')[1]
    assert d_cat == int(n_lat * DEC['n_cat_frac'])
    d_dec = shp('decoder.px_scale_decoder.0.weight')[1]
    assert shp('decoder.px_decoder.encoder.layers.0.block.0.weight') == (d_dec, n_lat + d_cat)
    assert len([k for k in sd if k.startswith('decoder.px_decoder.layers.') and k.endswith('ff.1.weight')]) \
        == DEC['n_layers']
    d_eff = shp('efficiency_head.net.0.weight')[0]
    banks = [('text', '_atom_text')] + [(Path(n).stem, f'_atom_prior_{i}')
                                        for i, n in enumerate(CLS['extra_src_names'])]
    bank_txt = ', '.join(f'{BANK_NAME.get(n, n)} ({shp("cls_emb." + b)[1]:,}-d)' for n, b in banks)
    n_train = int(D['cache_indices']['train'].split('=')[1].rstrip('>'))
    n_val = int(D['cache_indices']['val'].split('=')[1].rstrip('>'))
    assert n_train + n_val == reg['X']['summary_stats']['n_cells']
    learned = {'encoder': 'z_encoder.', 'decoder': ('decoder.cat_embeddings', 'decoder.px_decoder',
                                                    'decoder.px_scale_decoder', 'decoder.px_dropout_decoder', 'px_r'),
               'efficiency head': 'efficiency_head.', 'prototypes (trained rows)': None}
    npar = {}
    for k, pre in learned.items():
        if pre is None:
            npar[k] = n_lab * d_atom
        else:
            npar[k] = sum(v.numel() for n, v in sd.items() if n.startswith(pre) and v.is_floating_point()
                          and not any(s in n for s in ('ctrl_gene', 'ds_gene', 'pert_to_gene')))

    R = []

    def row(sec, name, val, key='', note=''):
        R.append(dict(section=sec, parameter=name, value=fmt(val), config_key=key, note=note))

    sec = 'Data'
    row(sec, 'Training cells', n_train, 'data.train_size', f'{n_val:,} cells held out for validation')
    row(sec, 'Datasets', n_ds, 'model_setup.batch_key')
    row(sec, 'Trained perturbations', n_lab, 'model_setup.labels_key')
    row(sec, 'Perturbation vocabulary (prototype rows)', n_vocab, '', f'{n_vocab - n_lab:,} rows filled zero-shot')
    row(sec, 'Genes (input and reconstruction)', n_genes)
    row(sec, 'Batch size (cells)', D['batch_size'], 'data.batch_size')
    row(sec, 'Batches per epoch', D['n_samples'], 'data.n_samples')
    row(sec, 'Cell sampling', 'weighted, with replacement', 'data.loader_type, use_balanced_weights, weights_mode',
        'weight ∝ (n + 1)^-1/2, n = cells of the same perturbation in the same dataset')
    sec = 'Input'
    row(sec, 'Transform', 'log(1 + x)', 'model.log_variational')
    row(sec, 'Centring', 'per-dataset control mean subtracted', 'encoder.center_input, center_mode',
        'control cells of each dataset; reconstruction target stays the raw counts')
    row(sec, 'Input noise (training)', f'multiplicative log-normal, σ = {fmt(E["noise_sigma"])}', 'encoder.noise_sigma')
    row(sec, 'Library size', 'observed (log total counts)')
    sec = 'Encoder'
    row(sec, 'Input layer', f'{n_genes:,} → {n_hid:,}', 'model.n_hidden', 'Linear, LayerNorm, GELU')
    row(sec, 'Residual blocks', E['n_layers'], 'encoder.n_layers', f'width {n_hid:,}; Gaussian activation noise σ = 0.01 (training)')
    row(sec, 'Dropout', M['dropout_rate'], 'model.dropout_rate')
    row(sec, 'Latent dimension', n_lat, 'model.n_latent', 'Gaussian posterior; sampled in training, mean at evaluation')
    row(sec, 'Latent centring (for scoring)', f'per-dataset EMA mean subtracted (momentum {0.05:g})',
        'model.invariant_align')
    sec = 'Decoder'
    row(sec, 'Dataset embedding', d_cat, 'decoder.n_cat_frac', f'concatenated to the latent ({DEC["n_cat_frac"]:g} × latent)')
    row(sec, 'Input layer', f'{n_lat + d_cat:,} → {d_dec:,}', '', 'Linear, LayerNorm, GELU')
    row(sec, 'Residual blocks', DEC['n_layers'], 'decoder.n_layers', f'width {d_dec:,}; no dropout')
    row(sec, 'Likelihood', 'zero-inflated negative binomial', 'model.gene_likelihood',
        'gene-wise dispersion; mean = softmax output × observed library')
    sec = 'Perturbation prototypes'
    row(sec, 'Prototype table', f'{n_vocab:,} × {d_atom:,}', 'cls_emb.random_init_proxies',
        'free parameters, initialised N(0, 0.02²); no adapter; cosine to the centred latent')
    sec = 'Objective'
    row(sec, 'Reconstruction weight', rl, 'anneal.rl_weight', 'ZINB negative log-likelihood, mean over cells')
    row(sec, 'KL weight', f'0 → {fmt(kl_last)} (linear)', 'anneal.kl_weight',
        f'ramp to {fmt(float(kl["max_weight"]))} over {P["n_epochs_warmup"]} epochs, stopped at {n_ep}')
    row(sec, 'Proxy Anchor weight', w_pa, 'model.proxy_anchor_weight', 'cells vs the trained prototypes')
    row(sec, 'Proxy Anchor scale α', M['proxy_anchor_alpha'], 'model.proxy_anchor_alpha')
    row(sec, 'Proxy Anchor margin δ', M['proxy_anchor_delta'], 'model.proxy_anchor_delta')
    row(sec, 'Proxy Anchor cell weight', 'efficiency rank in [0.1, 1]', 'model.use_efficiency_pa_weight',
        'within-batch, per-dataset rank of the measured perturbation efficiency')
    row(sec, 'Prior pull weight', w_pr, 'model.prior_atom_weight', f'ramped in linearly over {M["prior_atom_ramp_epochs"]} epochs')
    row(sec, 'Prior pull: pulled prototypes', f'lowest {100 * M["prior_atom_pull_q"]:g}% of evidence ({n_pulled:,})',
        'model.prior_atom_pull_q, prior_atom_evidence', 'evidence = rank of training cells × datasets')
    row(sec, 'Prior pull: target', f'mean of {M["prior_atom_graph_k"]} nearest prior neighbours',
        'model.prior_atom_map, prior_atom_graph_k',
        'cosine-weighted mean of well-supported prototypes, summed over the prior banks; detached')
    row(sec, 'Prior pull: banks', bank_txt, 'model_setup.class_emb_path, model.class_emb_extra_paths')
    row(sec, 'Prior pull: cross-fitting', f'{M["prior_atom_folds"]} folds, ≥ {M["prior_atom_min_fit"]} fit prototypes',
        'model.prior_atom_folds, prior_atom_min_fit', 'refit every epoch')
    row(sec, 'Prior pull: loss', 'squared distance of unit vectors', '', '2(1 − cos) to the target')
    row(sec, 'Prior pull: runaway guard', f'off-diagonal cosine > {fmt(M["prior_atom_coh_max"])}',
        'model.prior_atom_coh_max', 'disables the pull if exceeded; never tripped')
    row(sec, 'Zero-shot negative weight', w_ng, 'model.prior_atom_neg_weight',
        f'{M["prior_atom_neg_k"]} random untrained prototypes per step, filled by the same map; Proxy Anchor negative term')
    row(sec, 'Efficiency head weight', lam_eff, 'anneal.lambda_eff',
        f'MLP {n_lat:,} → {d_eff:,} → 1 on the detached latent; regresses the per-dataset efficiency rank')
    sec = 'Optimisation'
    row(sec, 'Optimiser', 'AdamW', '', 'ε = 0.01')
    row(sec, 'Learning rate', P['lr'], 'plan.lr', 'constant')
    row(sec, 'Weight decay', f"{P['weight_decay']:.0e}", 'plan.weight_decay')
    row(sec, 'Gradient clipping (norm)', T['gradient_clip_val'], 'train.gradient_clip_val')
    row(sec, 'Epochs', n_ep, 'train.max_epochs', f'{n_steps:,} steps; no early stop')
    row(sec, 'Precision', f'{T["precision"]}-bit', 'train.precision')
    sec = 'Zero-shot fill (shipped)'
    row(sec, 'Untrained rows', n_vocab - n_lab, '', f'mean of {M["prior_atom_graph_k"]} nearest prior neighbours, weights cos⁴, summed over the banks')
    sec = 'Learned parameters'
    for k, v in npar.items():
        row(sec, k.capitalize(), v)
    row(sec, 'Total', sum(npar.values()))

    T10 = pd.DataFrame(R)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    T10.to_csv(out / 'Table_S4.csv', index=False)
    with pd.ExcelWriter(out / 'Table_S4.xlsx') as w:
        T10.to_excel(w, sheet_name='S4 hyperparameters', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
    print(T10.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hparams', required=True, help='run_0/config/ExPert_v0/hparams.yaml')
    ap.add_argument('--model', required=True, help='run_0/config/ExPert_v0/model/model.pt')
    ap.add_argument('--events', required=True, help='the run\'s events.out.tfevents.*')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
