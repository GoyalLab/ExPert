# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.
"""Lossless compression of a SET of ExPert checkpoints for distribution.

WHY THIS EXISTS. A `model.pt` is ~2.1 GB, but most of it is not the model: the text bank
is stored three times (``cls_emb._atom_text``, ``registry_.setup_args.ext_df`` and the
``init_params_`` kwargs), the prior banks twice, ``class_centers`` is all zeros, and the
train / val / test indices are ~170 MB of int64. Every checkpoint trained from the same
banks carries the same ~1.1 GB, byte for byte. A fold sweep of 12 models is 25 GB of
which ~10 GB is unique.

WHAT IT DOES. ``pack`` walks every checkpoint's object tree and replaces large leaves
(tensors, numpy arrays, DataFrames >= ``min_bytes``):

    shared    a leaf whose content appears in >= 2 of the packed checkpoints (or twice in
              one) -> stored ONCE in ``shared.pt``, referenced by digest
    zeros     an all-zero tensor / array -> shape + dtype only
    zlib      an integer array / tensor (indices) -> zlib-compressed bytes

Everything else stays inline in ``<name>.slim.pt``. ``unpack`` rebuilds the original object
tree and writes a standard ``model.pt``, so ``ExPert.load`` / ``predict()`` need no change.
``verify`` compares two checkpoints leaf by leaf (exact equality, dtype, shape).

LOSSLESS, NOT BYTE-IDENTICAL. The restored object tree equals the original exactly, but
``torch.save`` output is not guaranteed to be byte-identical, and tensors that shared one
storage in the original are restored as separate (equal) tensors. Check with ``verify``,
never with md5 of the files.

    python -m expert.utils.ckpt_pack pack --out-dir packed fold0=.../model.pt fold1=.../model.pt
    python -m expert.utils.ckpt_pack unpack --slim packed/fold0.slim.pt --shared packed/shared.pt --out model.pt
    python -m expert.utils.ckpt_pack verify original/model.pt model.pt
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

__all__ = ['pack', 'unpack', 'verify']

TAG = '__ckpt_pack__'
FORMAT = 1


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _nbytes(x) -> int:
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, pd.DataFrame):
        return int(x.memory_usage(index=True, deep=False).sum())
    return 0


def _digest(x) -> str:
    h = hashlib.blake2b(digest_size=20)
    if torch.is_tensor(x):
        t = x.detach().contiguous()
        h.update(f'T|{t.dtype}|{tuple(t.shape)}'.encode())
        h.update(t.view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b'')
    elif isinstance(x, np.ndarray):
        h.update(f'A|{x.dtype.str}|{x.shape}'.encode())
        h.update(np.ascontiguousarray(x).tobytes() if x.dtype != object else repr(x.tolist()).encode())
    else:
        h.update(f'D|{x.shape}|{list(map(str, x.dtypes))}'.encode())
        h.update(repr(list(x.columns)).encode()); h.update(repr(list(x.index)).encode())
        v = x.to_numpy()
        h.update(np.ascontiguousarray(v).tobytes() if v.dtype != object else repr(v.tolist()).encode())
    return h.hexdigest()


def _is_zero(x) -> bool:
    if torch.is_tensor(x):
        return x.numel() > 0 and not x.is_complex() and not bool(x.detach().any())
    if isinstance(x, np.ndarray) and x.dtype != object:
        return x.size > 0 and not x.any()
    return False


def _is_int(x) -> bool:
    if torch.is_tensor(x):
        return not x.is_floating_point() and not x.is_complex() and x.dtype != torch.bool
    return isinstance(x, np.ndarray) and x.dtype.kind in 'iu'


def _leaves(obj, min_bytes, path=()):
    """yield (path, leaf) for every large tensor / array / DataFrame in the tree"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, min_bytes, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _leaves(v, min_bytes, path + (i,))
    elif _nbytes(obj) >= min_bytes:
        yield path, obj


def _map(obj, fn, min_bytes):
    """rebuild the tree, containers keep their type (OrderedDict, defaultdict, ... keep their type)"""
    if isinstance(obj, dict) and obj.get(TAG) is None:
        out = copy.copy(obj)                     # keeps the subclass and its state (defaultdict factory, ...)
        for k, v in obj.items():
            out[k] = _map(v, fn, min_bytes)
        return out
    if isinstance(obj, list):
        return [_map(v, fn, min_bytes) for v in obj]
    if isinstance(obj, tuple):
        vals = [_map(v, fn, min_bytes) for v in obj]
        return type(obj)(*vals) if hasattr(obj, '_fields') else tuple(vals)
    return fn(obj)


def _encode(x, shared_digests, shared_store, min_bytes, stats):
    if _nbytes(x) < min_bytes:
        return x
    if _is_zero(x):
        stats['zeros'] += _nbytes(x)
        if torch.is_tensor(x):
            return {TAG: 'zeros', 'kind': 'tensor', 'dtype': str(x.dtype).replace('torch.', ''),
                    'shape': list(x.shape), 'requires_grad': bool(x.requires_grad)}
        return {TAG: 'zeros', 'kind': 'array', 'dtype': x.dtype.str, 'shape': list(x.shape)}
    d = _digest(x)
    if d in shared_digests:
        shared_store.setdefault(d, x)
        stats['shared'] += _nbytes(x)
        return {TAG: 'ref', 'digest': d}
    if _is_int(x):
        stats['zlib'] += _nbytes(x)
        if torch.is_tensor(x):
            a = x.detach().contiguous().numpy()
            return {TAG: 'zlib', 'kind': 'tensor', 'dtype': a.dtype.str, 'shape': list(a.shape),
                    'requires_grad': False, 'data': zlib.compress(a.tobytes(), 6)}
        a = np.ascontiguousarray(x)
        return {TAG: 'zlib', 'kind': 'array', 'dtype': a.dtype.str, 'shape': list(a.shape),
                'data': zlib.compress(a.tobytes(), 6)}
    stats['inline'] += _nbytes(x)
    return x


def _decode(x, shared):
    if not (isinstance(x, dict) and TAG in x):
        return x
    kind = x[TAG]
    if kind == 'ref':
        return shared[x['digest']]
    if kind == 'zeros':
        if x['kind'] == 'tensor':
            return torch.zeros(x['shape'], dtype=getattr(torch, x['dtype'])).requires_grad_(x['requires_grad'])
        return np.zeros(x['shape'], dtype=np.dtype(x['dtype']))
    if kind == 'zlib':
        a = np.frombuffer(zlib.decompress(x['data']), dtype=np.dtype(x['dtype'])).reshape(x['shape']).copy()
        return torch.from_numpy(a) if x['kind'] == 'tensor' else a
    raise ValueError(f'unknown {TAG} entry {kind!r}')


def pack(checkpoints: dict[str, str], out_dir: str, min_bytes: int = 1_000_000) -> dict:
    """Pack ``{name: model.pt}`` into ``out_dir``/{name}.slim.pt + shared.pt + pack.json.

    Two passes so only one checkpoint is in memory at a time (plus the shared leaves)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    count = Counter()
    for name, p in checkpoints.items():
        ck = _load(p)
        for _, leaf in _leaves(ck, min_bytes):
            if not _is_zero(leaf):
                count[_digest(leaf)] += 1
        del ck
        log.info('digested %s', name)
    shared_digests = {d for d, c in count.items() if c >= 2}
    shared_store, report = {}, {}
    for name, p in checkpoints.items():
        ck = _load(p)
        stats = Counter()
        slim = _map(ck, lambda x: _encode(x, shared_digests, shared_store, min_bytes, stats), min_bytes)
        sp = out / f'{name}.slim.pt'
        torch.save({TAG: FORMAT, 'checkpoint': slim}, sp)
        report[name] = dict(source=str(p), source_bytes=Path(p).stat().st_size, slim_bytes=sp.stat().st_size,
                            **{f'{k}_bytes': int(v) for k, v in stats.items()})
        log.info('%s: %.2f GB -> %.2f GB (shared %.2f, zeros %.2f, zlib %.2f GB)', name,
                 Path(p).stat().st_size / 1e9, sp.stat().st_size / 1e9, stats['shared'] / 1e9,
                 stats['zeros'] / 1e9, stats['zlib'] / 1e9)
        del ck, slim
    torch.save({TAG: FORMAT, 'leaves': shared_store}, out / 'shared.pt')
    meta = dict(format=FORMAT, min_bytes=min_bytes, shared_bytes=(out / 'shared.pt').stat().st_size,
                n_shared=len(shared_store), checkpoints=report)
    (out / 'pack.json').write_text(json.dumps(meta, indent=1))
    return meta


def unpack(slim_path: str, shared_path: str, out_path: str | None = None, shared: dict | None = None):
    """Rebuild a checkpoint; written to ``out_path`` when given, returned either way.
    Pass an already-loaded ``shared`` dict to restore many checkpoints without re-reading it."""
    s = _load(slim_path)
    if not (isinstance(s, dict) and s.get(TAG) == FORMAT):
        raise ValueError(f'{slim_path} is not a ckpt_pack v{FORMAT} file')
    if shared is None:
        shared = _load(shared_path)['leaves']
    ck = _map(s['checkpoint'], lambda x: _decode(x, shared), 0)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, out_path)
    return ck


def _equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and a.requires_grad == b.requires_grad \
            and bool(torch.equal(a, b) or (a.is_floating_point() and torch.equal(a.isnan(), b.isnan())
                                          and torch.equal(a.nan_to_num(), b.nan_to_num())))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    if isinstance(a, pd.DataFrame):
        return a.equals(b) and a.index.equals(b.index) and a.columns.equals(b.columns)
    if isinstance(a, dict):
        return list(a) == list(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    try:
        r = a == b
        return bool(r) if not isinstance(r, (np.ndarray, pd.Series, pd.DataFrame)) else bool(np.all(r))
    except Exception:
        return repr(a) == repr(b)


def verify(a, b) -> list[str]:
    """Paths (as strings) where two checkpoints differ; empty = identical object trees.
    ``a`` / ``b`` are paths or already-loaded checkpoints."""
    a = _load(a) if isinstance(a, (str, Path)) else a
    b = _load(b) if isinstance(b, (str, Path)) else b
    bad = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict) and type(x) is type(y):
            if list(x) != list(y):
                bad.append(f'{path}: keys differ')
            for k in x:
                if k in y:
                    walk(x[k], y[k], f'{path}.{k}' if path else str(k))
        elif not _equal(x, y):
            bad.append(path or '<root>')

    walk(a, b, '')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack', help='name=model.pt ... -> out-dir')
    p.add_argument('checkpoints', nargs='+', help='name=path/to/model.pt')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--min-bytes', type=int, default=1_000_000)
    p.add_argument('--verify', action='store_true', help='unpack every checkpoint and compare with its source')
    u = sub.add_parser('unpack')
    u.add_argument('--slim', required=True)
    u.add_argument('--shared', required=True)
    u.add_argument('--out', required=True)
    v = sub.add_parser('verify')
    v.add_argument('a'); v.add_argument('b')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if a.cmd == 'pack':
        cks = dict(x.split('=', 1) for x in a.checkpoints)
        meta = pack(cks, a.out_dir, a.min_bytes)
        tot = sum(r['source_bytes'] for r in meta['checkpoints'].values())
        new = sum(r['slim_bytes'] for r in meta['checkpoints'].values()) + meta['shared_bytes']
        log.info('packed %d checkpoints: %.1f GB -> %.1f GB', len(cks), tot / 1e9, new / 1e9)
        if a.verify:
            shared = _load(Path(a.out_dir) / 'shared.pt')['leaves']
            fails = {}
            for name, src in cks.items():
                d = verify(src, unpack(Path(a.out_dir) / f'{name}.slim.pt', None, shared=shared))
                fails[name] = d
                log.info('verify %s: %s', name, 'IDENTICAL' if not d else f'{len(d)} differences, e.g. {d[:3]}')
            meta['verified'] = {k: not v for k, v in fails.items()}
            (Path(a.out_dir) / 'pack.json').write_text(json.dumps(meta, indent=1))
            if any(fails.values()):
                raise SystemExit(1)
    elif a.cmd == 'unpack':
        unpack(a.slim, a.shared, a.out)
    else:
        d = verify(a.a, a.b)
        print('IDENTICAL' if not d else '\n'.join(d))
        raise SystemExit(1 if d else 0)


if __name__ == '__main__':
    main()
