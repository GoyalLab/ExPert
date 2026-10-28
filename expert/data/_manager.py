# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

from scvi.data._manager import AnnDataManager, AnnDataManagerValidationCheck
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data._utils import scipy_to_torch_sparse
from torch.utils.data import Subset
import os
import atexit
import fcntl
import mmap
import hashlib
import torch
import h5py
from scipy.sparse import issparse, csr_matrix
from anndata.abc import CSCDataset, CSRDataset
import pandas as pd
import numpy as np

from expert.utils.constants import REGISTRY_KEYS
from collections.abc import Sequence
from scvi.data.fields import AnnDataField

SparseDataset = (CSRDataset, CSCDataset)

import logging


def _slice_csr_rows(mat: csr_matrix, rows: np.ndarray | list | slice, dtype=None):
    """Slice CSR rows via indptr without scipy's internal int64 upcast."""
    if isinstance(rows, slice):
        rows = np.arange(*rows.indices(mat.shape[0]))
    rows = np.asarray(rows)
    indptr = mat.indptr
    starts = indptr[rows]
    ends = indptr[rows + 1]
    lengths = ends - starts
    new_nnz = int(lengths.sum())
    new_indptr = np.empty(len(rows) + 1, dtype=indptr.dtype)
    new_indptr[0] = 0
    np.cumsum(lengths, out=new_indptr[1:])
    new_data = np.empty(new_nnz, dtype=dtype or mat.data.dtype)
    new_indices = np.empty(new_nnz, dtype=mat.indices.dtype)
    pos = 0
    for s, l in zip(starts, lengths):
        new_data[pos:pos + l] = mat.data[s:s + l]
        new_indices[pos:pos + l] = mat.indices[s:s + l]
        pos += l
    return csr_matrix((new_data, new_indices, new_indptr), shape=(len(rows), mat.shape[1]))


SHM_DIR = "/dev/shm"
SHM_PREFIX = "expert_x_"


def _csr_identity(X) -> str:
    """Content-derived key for a CSR matrix, stable across processes.

    Two jobs reading the SAME database must land on the SAME key, and two jobs reading different
    databases must not. Keying on the file PATH is not an option here: this path only runs for
    in-memory (non-backed) adata, where ``adata.filename`` is None. So the key is derived from the
    structure itself -- shape, nnz, dtypes, the full indptr (which fixes the row structure), and
    the head/tail of data+indices. Hashing indptr costs ~0.1s at 19.5M rows and makes an accidental
    collision between two different databases implausible.
    """
    h = hashlib.sha1()
    h.update(repr((X.shape, int(X.nnz), str(X.data.dtype),
                   str(X.indices.dtype), str(X.indptr.dtype))).encode())
    h.update(np.ascontiguousarray(X.indptr).tobytes())
    # STRIDED, not just the ends. Same ~2MB of reads per array, but spread over its whole length:
    # end-sampling alone would miss two matrices that share structure and differ only in the
    # middle -- e.g. the same database regenerated with a different normalisation. Cheap insurance
    # on a 44.5GB `data` array; a collision means two jobs would silently train on each other's X.
    CHUNKS, CHUNK = 32, 1 << 16                       # 32 x 64KB = 2MB per array
    for arr in (X.data, X.indices):
        a = np.ascontiguousarray(arr)
        n = a.shape[0]
        if n <= CHUNKS * CHUNK:
            h.update(a.tobytes())                     # small enough to hash outright
            continue
        step = n // CHUNKS
        for i in range(CHUNKS):
            off = i * step
            h.update(a[off:off + CHUNK].tobytes())
        h.update(a[-CHUNK:].tobytes())                # always include the tail
    return h.hexdigest()[:16]


def _shm_attach_or_create(key: str, X, attrs, log) -> tuple[dict, object]:
    """Attach to a node-wide shm copy of the CSR buffers, creating it if we are first.

    WHY THIS EXISTS. ``torch.Tensor.share_memory_()`` mints an ANONYMOUS, PID-prefixed segment
    (observed: ``torch_1368698_673948381_13488``), so every job allocates its OWN copy of identical
    bytes. Three v0.1.3 arms on one node each relocated 133.7GB of the same database -- 401GB of
    duplication -- and exhausted a node-wide tmpfs that ``--mem`` does not protect.

    TWO LOCKS, DELIBERATELY. A single lock deadlocks: if the same file is used to serialise
    creation (exclusive) AND to mark "I am attached" (shared, held for the run), then the second
    job's exclusive acquire blocks until the first job EXITS -- which is precisely the case this
    function exists to serve. Caught by test 2, which hung for 15 minutes.
      * ``.cflock`` -- creation lock. Exclusive, held only for the check-and-build, then released.
      * ``.rlock``  -- reference lock. Shared, held for the process lifetime; its exclusive
                       acquire succeeding is what tells the cleanup it is the last user.

    Returns the numpy views and the reference-lock file object, which the CALLER MUST KEEP ALIVE --
    dropping it releases the shared lock and invites a sibling to delete the segment underneath.
    """
    base = os.path.join(SHM_DIR, f"{SHM_PREFIX}{key}")
    # Sizes WITHOUT materialising anything: .nbytes/.dtype/.shape do not copy.
    want = {n: (getattr(X, n).nbytes, getattr(X, n).dtype, getattr(X, n).shape) for n in attrs}
    total_gb = sum(nb for nb, _, _ in want.values()) / 1024 ** 3

    cflock = open(base + ".cflock", "a+")
    try:
        fcntl.flock(cflock, fcntl.LOCK_EX)          # brief: only around check-and-build
        ready = os.path.exists(base + ".ready")
        if ready:
            ok = all(os.path.exists(f"{base}.{n}") and os.path.getsize(f"{base}.{n}") == nb
                     for n, (nb, _, _) in want.items())
            if not ok:                              # torn segment from a killed creator
                log.warning("[shared-mem] stale/incomplete segment %s -- rebuilding", base)
                for n in list(want) + ["ready"]:
                    try: os.unlink(f"{base}.{n}")
                    except OSError: pass
                ready = False
        views, maps = {}, []
        for n, (nb, dt, shape) in want.items():
            fd = os.open(f"{base}.{n}", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if not ready:
                    os.ftruncate(fd, nb)
                mm = mmap.mmap(fd, nb, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)                        # mmap holds its own reference
            v = np.ndarray(shape, dtype=dt, buffer=mm)
            if not ready:
                # FREE AS WE GO. Holding all three private buffers across the whole relocation
                # makes the high-water mark need+need (249GB at full scale) instead of
                # need+largest (207GB) -- that regression OOM-killed v0.1.3 arm A2 at 250G even
                # though the /dev/shm preflight passed. Rebinding X here drops the private buffer
                # for THIS array before the next one is copied, which is what the original
                # share_memory_() loop did.
                src = np.ascontiguousarray(getattr(X, n))
                v[:] = src
                del src
            setattr(X, n, v)                        # rebind now; private buffer released
            views[n] = v
            maps.append(mm)
        if not ready:
            open(base + ".ready", "w").close()
            log.info("[shared-mem] CREATED node-wide segment %s (%.1fGB) -- co-resident jobs on "
                     "this database attach instead of copying", base, total_gb)
        else:
            log.info("[shared-mem] ATTACHED to existing segment %s (%.1fGB) -- no copy made",
                     base, total_gb)
    finally:
        fcntl.flock(cflock, fcntl.LOCK_UN)
        cflock.close()

    rlock = open(base + ".rlock", "a+")
    fcntl.flock(rlock, fcntl.LOCK_SH)               # held for the process lifetime
    rlock._expert_maps = maps                       # keep the mmaps alive with the lock
    rlock._expert_base = base
    return views, rlock


def _shm_release(lock) -> None:
    """Last user on the node unlinks the segment; otherwise leave it for the others."""
    if lock is None:
        return
    base = getattr(lock, "_expert_base", None)
    log = logging.getLogger(__name__)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.info("[shared-mem] siblings still attached to %s -- leaving it", base)
            return
        for suf in ("data", "indices", "indptr", "ready"):
            try: os.unlink(f"{base}.{suf}")
            except OSError: pass
        log.info("[shared-mem] last user on node -- removed %s", base)
    finally:
        # MUST release and close. The probe above UPGRADES this fd's lock from shared to
        # exclusive, and .rlock is never unlinked (unlinking a live flock target is a race), so
        # leaving it held means the next LOCK_SH on that same inode blocks forever -- a later
        # attach to the same key in this process would deadlock. Caught by test 5, which hung.
        try: fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError: pass
        try: lock.close()
        except OSError: pass
        lock._expert_maps = None


def _maybe_share_x_memory(manager: AnnDataManager) -> None:
    """Move the in-memory sparse X into POSIX shared memory (``/dev/shm``), ONCE.

    Rationale: with ``num_workers>0`` the DataLoader forks worker processes that
    map the parent's private-heap CSR arrays copy-on-write. Python refcount churn
    and malloc/GC bookkeeping on pages *adjacent* to those arrays dirties them, so
    each worker slowly faults in its own private copy — measured ~40% duplication
    on the full run (435GB peak vs 310GB sparse). Relocating the three CSR buffers
    (``data``/``indices``/``indptr``) into a ``MAP_SHARED`` mmap segment means all
    forked workers share ONE physical copy (reads never dirty shared pages), so RAM
    stays at ~1x sparse while ``num_workers`` (throughput) is untouched.

    NODE-WIDE AS OF 2026-09-06. The segment is now named deterministically from the CSR content
    (``_csr_identity``) and guarded by a flock, so jobs reading the SAME database ATTACH to one
    copy instead of each minting their own. Previously ``share_memory_()`` produced an anonymous
    PID-prefixed segment (``torch_<pid>_<rand>_<n>``), so three co-resident v0.1.3 arms each
    relocated the same 133.7GB -- 401GB of duplicated identical bytes -- and exhausted /dev/shm,
    which is node-wide and NOT covered by --mem. Only the first job pays the copy; the rest attach.

    Idempotent: rebinds ``adata.X``'s buffers *in place* to shared-memory-backed
    numpy views, so every dataset that references the same ``adata.X`` (train/val/
    test) — and every existing ``.toarray()`` slice — reads from the one shared copy
    with no other code change. Must run BEFORE datasets capture ``adata.X``.

    Peak note: the copy is done free-as-you-go -- each buffer is mmap'd, filled from
    its private original, and rebound before the next one starts -- so the ``/dev/shm``
    high-water mark is exactly the sparse size and the transient second copy lives in
    HOST ram, not in the tmpfs. No-op for backed adata.
    """
    if getattr(manager, "_shared_x_done", False):
        return
    X = manager.adata.X
    if not (issparse(X) and getattr(X, "format", None) == "csr"):
        return

    # FAIL LOUD, NOT LATE. /dev/shm is a NODE-WIDE tmpfs shared by every job on the box and is NOT
    # protected by SLURM --mem, so co-resident arms can exhaust it even when each is individually
    # within budget. Without this check the relocation half-succeeds and the failure surfaces later
    # as `RuntimeError: unable to write to file </torch_...>: No space left on device (28)` raised
    # inside a forked DataLoader WORKER -- which the main process does not treat as fatal. Measured
    # cost of that: v0.1.3 arms A2 and P logged 1.4h and 0.9h of training and then hung for the
    # remainder of a 12h wall clock, producing 10 and 6 epochs against N's 60.
    _log = logging.getLogger(__name__)
    key = _csr_identity(X)
    _sizes = {a: int(getattr(X, a).nbytes) for a in ("data", "indices", "indptr")}
    # /dev/shm high-water mark is exactly the sparse size: each buffer is mmap'd, filled from its
    # private original and rebound before the next starts, so the transient second copy is in HOST
    # ram. An ATTACH to an already-populated segment costs nothing at all, so skip the check then.
    required = need = sum(_sizes.values())
    if os.path.exists(os.path.join(SHM_DIR, f"{SHM_PREFIX}{key}") + ".ready"):
        required = 0
    try:
        _st = os.statvfs("/dev/shm")
        free = _st.f_bavail * _st.f_frsize
        total = _st.f_blocks * _st.f_frsize
    except OSError:
        free = total = None
    G = 1024 ** 3
    if free is not None and free < required:
        raise RuntimeError(
            f"shared_mem_x needs {required / G:.1f}GB free in /dev/shm "
            f"(CSR buffers: data {_sizes['data'] / G:.1f} + "
            f"indices {_sizes['indices'] / G:.1f} + indptr {_sizes['indptr'] / G:.2f}), "
            f"but only {free / G:.1f}GB of "
            f"{total / G:.1f}GB is free on this node. /dev/shm is shared across ALL jobs on the "
            f"node and is not covered by --mem, so a co-resident run is most likely holding it. "
            f"Either give this arm a node to itself, or set data.shared_mem_x=false and raise "
            f"cluster.mem_gb (expect ~40% more RAM from copy-on-write duplication across "
            f"num_workers)."
        )
    if free is not None:
        logging.getLogger(__name__).info(
            "[shared-mem] /dev/shm preflight OK: need %.1fGB (%s), free %.1fGB of %.1fGB",
            need / G, "attach, 0GB new" if required == 0 else "create", free / G, total / G)
    # X is passed through, NOT a pre-materialised dict: the helper rebinds each buffer as soon as
    # it is copied, so only ONE private array is live at a time (peak need+largest, not need+need).
    views, lock = _shm_attach_or_create(key, X, ("data", "indices", "indptr"), _log)
    # Keep the lock (and, through it, the mmaps) alive for the process lifetime: releasing the
    # shared lock tells the last-exiter cleanup that nobody is attached, and it would unlink the
    # segment out from under a still-running sibling.
    manager._shared_x_lock = lock
    manager._shared_x_done = True
    atexit.register(_shm_release, lock)
    logging.getLogger(__name__).info(
        "[shared-mem] X CSR relocated to /dev/shm: data=%.1fGB indices=%.1fGB indptr=%.2fGB "
        "-> forked workers now mmap one shared copy (num_workers unchanged)",
        X.data.nbytes / 1e9, X.indices.nbytes / 1e9, X.indptr.nbytes / 1e9,
    )


class EmbAnnTorchDataset(AnnTorchDataset):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
        ignore_types: list = ['uns', 'varm'],
    ):
        super().__init__(adata_manager, getitem_tensors, load_sparse_tensor)
        self.ignore_types = ignore_types

    def __getitem__(
        self, indexes: int | list[int] | slice
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Fetch data from the :class:`~anndata.AnnData` object.

        Parameters
        ----------
        indexes
            Indexes of the observations to fetch. Can be a single index, a list of indexes, or a
            slice.

        Returns
        -------
        Mapping of data registry keys to arrays of shape ``(n_obs, ...)``.
        """
        if isinstance(indexes, int):
            indexes = [indexes]  # force batched single observations

        if self.adata_manager.adata.isbacked and isinstance(indexes, list | np.ndarray):
            # need to sort indexes for h5py datasets
            indexes = np.sort(indexes)

        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]
            idx_slice = indexes
            if self.adata_manager.data_registry[key].attr_name in self.ignore_types:
                # Ignore .uns and .varm
                continue
            if isinstance(data, np.ndarray | h5py.Dataset):
                sliced_data = data[idx_slice].astype(dtype, copy=False)
            elif isinstance(data, pd.DataFrame):
                sliced_data = data.iloc[idx_slice, :].to_numpy().astype(dtype, copy=False)
            elif issparse(data) or isinstance(data, SparseDataset):
                if issparse(data) and data.format == 'csr' and data.indptr.dtype == np.int64:
                    sliced_data = _slice_csr_rows(data, idx_slice, dtype=dtype)
                else:
                    sliced_data = data[idx_slice].astype(dtype, copy=False)
                if self.load_sparse_tensor:
                    sliced_data = scipy_to_torch_sparse(sliced_data)
                else:
                    sliced_data = sliced_data.toarray()
            elif isinstance(data, str) and key == REGISTRY_KEYS.MINIFY_TYPE_KEY:
                # for minified anndata, we can have a string for `data`,
                # which is the value of the MINIFY_TYPE_KEY in adata.uns,
                # used to record the type data minification
                continue
            else:
                raise TypeError(f"{key} is not a supported type")

            data_map[key] = sliced_data

        return data_map


class EmbAnnDataManager(AnnDataManager):

    def __init__(
        self,
        fields: list[AnnDataField] | None = None,
        setup_method_args: dict | None = None,
        validation_checks: AnnDataManagerValidationCheck | None = None,
    ) -> None:
        super().__init__(fields, setup_method_args, validation_checks)

    def create_torch_dataset(
        self,
        indices: Sequence[int] | Sequence[bool] = None,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
    ) -> AnnTorchDataset:
        """
        Creates a torch dataset from the AnnData object registered with this instance.

        Parameters
        ----------
        indices
            The indices of the observations in the adata to use
        data_and_attributes
            Dictionary with keys representing keys in data registry
            (``adata_manager.data_registry``) and value equal to desired numpy loading type (later
            made into torch tensor) or list of such keys. A list can be used to subset to certain
            keys in the event that more tensors than needed have been registered. If ``None``,
            defaults to all registered data.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data.

        Returns
        -------
        :class:`~scvi.data.AnnTorchDataset`
        """
        # Relocate the shared sparse X into shared memory so forked workers mmap one
        # copy instead of COW-duplicating it (opt-in via data.shared_mem_x). Idempotent;
        # only meaningful for the in-RAM sparse path.
        if getattr(self, "_want_shared_x", False):
            _maybe_share_x_memory(self)

        dataset = EmbAnnTorchDataset(
            self,
            getitem_tensors=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )
        if indices is not None:
            # This is a lazy subset, it just remaps indices
            dataset = Subset(dataset, indices)
        return dataset
