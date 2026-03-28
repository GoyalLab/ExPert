"""Dense memory-mapped cache for fast random-access training on large AnnData.

One-time conversion from backed (sparse) AnnData to a dense .npy file.
At training time, np.load(mmap_mode='r') gives instant random row access
via OS page cache — no custom caching, no sparse overhead.

Usage:
    # One-time: convert backed adata to dense cache
    cache = DenseCache.create(adata, cache_path='./X_dense.npy')

    # Training: load as mmap, random access is ~1ms per batch
    cache = DenseCache(cache_path='./X_dense.npy')
    batch = cache[indices]  # returns np.ndarray (n, n_vars)
"""

import os
import json
import shutil
import logging
import subprocess
import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

# Track SSD cache paths created during this process for cleanup
_ssd_cache_paths: list[str] = []


def _is_on_gpfs(path: str) -> bool:
    """Check if a path resides on a GPFS filesystem."""
    path = os.path.realpath(path)
    if path.startswith('/gpfs'):
        return True
    try:
        result = subprocess.run(
            ['df', '--output=fstype', path],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2 and 'gpfs' in lines[-1].lower():
                return True
    except Exception:
        pass
    return False


def _find_local_ssd_scratch() -> str | None:
    """Find a local SSD scratch directory on the node.

    Looks for non-rotational (SSD/NVMe) block devices with local mount points
    that have usable scratch space. Returns the best candidate path or None.
    """
    try:
        # Get non-rotational block devices and their mount points
        result = subprocess.run(
            ['lsblk', '-ndo', 'NAME,ROTA'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        ssd_devices = set()
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) == 2 and parts[1] == '0':
                ssd_devices.add(parts[0])

        if not ssd_devices:
            return None

        # Get mount points for all partitions of SSD devices
        result = subprocess.run(
            ['lsblk', '-nro', 'NAME,MOUNTPOINT'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        ssd_mounts = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1]:
                dev_name = parts[0]
                mountpoint = parts[1]
                # Check if this partition belongs to an SSD device
                if any(dev_name.startswith(ssd) for ssd in ssd_devices):
                    # Skip boot/efi/swap partitions and network mounts
                    if mountpoint in ('/boot', '/boot/efi') or mountpoint.startswith('/gpfs'):
                        continue
                    ssd_mounts.append(mountpoint)

        if not ssd_mounts:
            return None

        # Prefer /tmp if it's on an SSD, otherwise use the root mount
        for candidate in ['/tmp'] + ssd_mounts:
            if candidate in ssd_mounts or (candidate == '/tmp' and '/' in ssd_mounts):
                if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
                    return candidate

        # Fall back to first writable SSD mount
        for mount in ssd_mounts:
            if os.path.isdir(mount) and os.access(mount, os.W_OK):
                return mount

    except Exception as e:
        logger.debug(f"Failed to detect local SSD scratch: {e}")

    return None


def try_migrate_cache_to_ssd(cache_path: str) -> str:
    """If cache is on GPFS and a local SSD has space, copy it there.

    Returns the (possibly new) cache path to use.
    """
    if not os.path.exists(cache_path):
        return cache_path

    if not _is_on_gpfs(cache_path):
        logger.debug("Cache not on GPFS, skipping SSD migration")
        return cache_path

    ssd_scratch = _find_local_ssd_scratch()
    if ssd_scratch is None:
        logger.info("No local SSD scratch found, using GPFS cache")
        return cache_path

    # Calculate cache size (npy + meta)
    meta_path = cache_path + '.meta.json'
    cache_size = os.path.getsize(cache_path)
    if os.path.exists(meta_path):
        cache_size += os.path.getsize(meta_path)

    # Check available space on SSD - need at least 2x cache size
    # (half the disk must remain free)
    disk_usage = shutil.disk_usage(ssd_scratch)
    if cache_size > disk_usage.free / 2:
        logger.info(
            f"Not enough SSD space for cache migration: "
            f"cache={cache_size / 1e9:.1f}GB, "
            f"available={disk_usage.free / 1e9:.1f}GB on {ssd_scratch}"
        )
        return cache_path

    # Build SSD cache path: <ssd_scratch>/dense_cache_<user>/<basename>
    user = os.environ.get('USER', 'unknown')
    ssd_cache_dir = os.path.join(ssd_scratch, f'dense_cache_{user}')
    os.makedirs(ssd_cache_dir, exist_ok=True)
    ssd_cache_path = os.path.join(ssd_cache_dir, os.path.basename(cache_path))
    ssd_meta_path = ssd_cache_path + '.meta.json'

    # Skip copy if already migrated (same size = already copied)
    if (
        os.path.exists(ssd_cache_path)
        and os.path.exists(ssd_meta_path)
        and os.path.getsize(ssd_cache_path) == os.path.getsize(cache_path)
    ):
        logger.info(f"SSD cache already exists at {ssd_cache_path}")
        return ssd_cache_path

    logger.info(
        f"Migrating dense cache to local SSD: "
        f"{cache_path} -> {ssd_cache_path} ({cache_size / 1e9:.1f}GB)"
    )
    try:
        shutil.copy2(cache_path, ssd_cache_path)
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, ssd_meta_path)
        logger.info(f"Cache migrated to SSD at {ssd_cache_path}")
        _ssd_cache_paths.append(ssd_cache_path)
        return ssd_cache_path
    except Exception as e:
        logger.warning(f"Failed to migrate cache to SSD: {e}, using GPFS cache")
        # Clean up partial copy
        for p in (ssd_cache_path, ssd_meta_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return cache_path


class DenseCache:
    """Memory-mapped dense array for fast random-access batch loading."""

    def __init__(self, cache_path: str):
        meta_path = cache_path + '.meta.json'
        with open(meta_path) as f:
            meta = json.load(f)

        self.cache_path = cache_path
        self.n_obs = meta['n_obs']
        self.n_vars = meta['n_vars']
        self.dtype = meta.get('dtype', 'float32')

        # Memory-map: no RAM used, OS page cache handles reads
        self._mmap = np.load(cache_path, mmap_mode='r')

    def __getitem__(self, indices) -> np.ndarray:
        """Fetch rows by index. Returns a dense copy (writable)."""
        return np.array(self._mmap[indices])

    @property
    def shape(self):
        return (self.n_obs, self.n_vars)

    @classmethod
    def create(
        cls,
        adata,
        cache_path: str,
        chunk_size: int = 50_000,
        dtype: str = 'int16',
    ) -> 'DenseCache':
        """Convert backed AnnData X to a dense .npy file.

        Reads in chunks to avoid loading the full sparse matrix at once.

        Parameters
        ----------
        adata
            AnnData object (backed or in-memory).
        cache_path
            Output .npy file path.
        chunk_size
            Rows to convert per chunk. Controls peak memory during conversion.
        dtype
            Output dtype.
        """
        n_obs, n_vars = adata.shape
        disk_size_gb = n_obs * n_vars * np.dtype(dtype).itemsize / 1e9
        logger.info(
            f"Creating dense cache: {n_obs:,} x {n_vars:,} = {disk_size_gb:.1f} GB at {cache_path}"
        )

        # Pre-allocate the full dense file on disk using mmap
        fp = np.lib.format.open_memmap(
            cache_path, mode='w+', dtype=dtype, shape=(n_obs, n_vars)
        )

        from tqdm import tqdm
        n_chunks = int(np.ceil(n_obs / chunk_size))
        for i in tqdm(range(n_chunks), desc="Converting to dense", unit="chunk"):
            start = i * chunk_size
            end = min(start + chunk_size, n_obs)
            chunk = adata.X[start:end]
            if issparse(chunk):
                chunk = chunk.toarray()
            fp[start:end] = np.asarray(chunk, dtype=dtype)

        # Flush to disk
        del fp

        # Write metadata
        meta = {'n_obs': n_obs, 'n_vars': n_vars, 'dtype': dtype}
        with open(cache_path + '.meta.json', 'w') as f:
            json.dump(meta, f)

        logger.info(f"Dense cache created: {n_obs:,} rows, {n_vars:,} cols, {disk_size_gb:.1f} GB")
        return cls(cache_path)


def cleanup_ssd_cache():
    """Remove any SSD cache files created during this process."""
    for cache_path in _ssd_cache_paths:
        meta_path = cache_path + '.meta.json'
        for p in (cache_path, meta_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
                    logger.info(f"Removed SSD cache file: {p}")
            except OSError as e:
                logger.warning(f"Failed to remove SSD cache file {p}: {e}")
        # Remove parent dir if empty
        parent = os.path.dirname(cache_path)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass
    _ssd_cache_paths.clear()
