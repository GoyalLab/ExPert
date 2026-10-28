# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.
"""Drop checkpoint `__init__` kwargs the current schema no longer accepts.

WHY THIS EXISTS. A checkpoint stores the exact `__init__` kwargs it was built with
(`init_params_`). scvi replays them verbatim on load. So the moment a parameter is
deleted from a signature, every checkpoint that recorded it becomes unloadable with a
bare `TypeError: __init__() got an unexpected keyword argument`. That makes any cleanup
of the kwarg surface a breaking change for saved models — which is precisely the thing
blocking the module refactor.

WHAT IT DOES. On the load path only, compare the recorded kwargs against the set the
CURRENT classes accept and drop the rest, loudly.

    accepted = (every named param anywhere in the module MRO)
             u (keys the MODEL consumes itself and never forwards)

Legacy knob families are translated to the current schema BEFORE this filter (expert.utils.legacy).

THE ERROR DIRECTION IS DELIBERATE. This filter is biased to KEEP. A false drop is silent
and changes what the model does; a false keep is a loud TypeError at construction. So the
MRO walk collects named params from every class in the chain rather than trying to track
exactly which `**kwargs` forwards where — a superset is the safe side.

LOAD PATH ONLY. This must never run during ordinary construction. A typo'd knob in a
fresh training config has to fail loudly instead of being silently dropped, which is
the whole reason `filter_init_kwargs` is gated behind a flag the loader sets rather than
being wired into `__init__` unconditionally.

WHAT IT DOES NOT DO. It does not touch `model.pt` — the filtering is in-memory, so a
checkpoint loaded by an older version of the code still behaves exactly as it did.
"""
from __future__ import annotations

import inspect
import logging

log = logging.getLogger(__name__)

__all__ = [
    'accepted_init_kwargs',
    'filter_init_kwargs',
]

# Cache: {(model_cls, module_cls): frozenset}
_ACCEPTED_CACHE: dict[tuple, frozenset] = {}


def _named_params(cls) -> set[str]:
    """Named (non-`self`, non-`*args`/`**kwargs`) parameters of ``cls.__init__``."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != 'self' and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def _mro_named_params(cls) -> set[str]:
    """Every named `__init__` parameter anywhere in ``cls``'s MRO.

    Deliberately a SUPERSET of what any single class accepts: the module forwards its
    `**vae_kwargs` up the chain, and mapping each key to the exact class that consumes it
    would be brittle. Over-accepting costs a loud TypeError; under-accepting silently
    drops a live knob. See the module docstring on error direction.
    """
    out: set[str] = set()
    for base in getattr(cls, '__mro__', [cls]):
        if base is object:
            continue
        out |= _named_params(base)
    return out


def accepted_init_kwargs(model_cls, module_cls=None) -> frozenset[str]:
    """The full set of `__init__` kwarg names the current schema accepts.

    Parameters
    ----------
    model_cls
        The model class (e.g. ``ExPert``). Read for its own named params, its
        ``_model_level_kwargs`` declaration, and — if ``module_cls`` is not given — its
        ``_module_cls`` attribute.
    module_cls
        The module class (e.g. ``XPert``). Defaults to ``model_cls._module_cls``.
    """
    if module_cls is None:
        module_cls = getattr(model_cls, '_module_cls', None)
    key = (model_cls, module_cls)
    hit = _ACCEPTED_CACHE.get(key)
    if hit is not None:
        return hit

    acc: set[str] = set()
    # 1. Everything the model itself binds by name.
    acc |= _mro_named_params(model_cls)
    # 2. Everything the module (and its parents, via **vae_kwargs) binds by name.
    if module_cls is not None:
        acc |= _mro_named_params(module_cls)
    # 3. Keys the MODEL pops/reads out of its own kwarg bag and never forwards. These
    #    appear in no signature at all, so without this they would all be dropped.
    acc |= {str(k) for k in getattr(model_cls, '_model_level_kwargs', ()) or ()}
    # 4. Same, for anything the module consumes without declaring.
    if module_cls is not None:
        acc |= {str(k) for k in getattr(module_cls, '_module_level_kwargs', ()) or ()}

    out = frozenset(acc)
    _ACCEPTED_CACHE[key] = out
    return out


def filter_init_kwargs(
    kwargs: dict,
    model_cls,
    module_cls=None,
    *,
    context: str = 'checkpoint',
) -> tuple[dict, list[str]]:
    """Return ``(kept, dropped)`` for a recorded kwarg bag.

    Pure function — does not mutate ``kwargs``. Every dropped key is logged at WARNING
    with the context, because a silently altered model is far worse than a noisy load.
    """
    accepted = accepted_init_kwargs(model_cls, module_cls)

    kept, dropped = {}, []
    for k, v in kwargs.items():
        if k in accepted:
            kept[k] = v
        else:
            dropped.append(k)

    if dropped:
        log.warning(
            f'[Compat] Dropped {len(dropped)} unsupported {context} param(s) not in the '
            f'current schema: {", ".join(sorted(dropped))}'
        )
    return kept, sorted(dropped)

