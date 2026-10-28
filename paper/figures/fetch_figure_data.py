# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Download the figures' plotting data from Zenodo and unpack it into the figure folders.

The Zenodo record "ExPert - figure data" (doi:10.5281/zenodo.23740198) holds one zip per figure,
figure_data_<fig>.zip, whose members are figures/<fig>/{data,inputs}/... . In this repository the
supplementary figures live under figures/si/, so figS* members are unpacked to si/figS*/ instead.
Every zip is checked against the md5 Zenodo lists before it is unpacked.

    python paper/figures/fetch_figure_data.py                  # all figures
    python paper/figures/fetch_figure_data.py --figs fig3 figS2
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RECORD = '23740198'


def target(fig: str) -> str:
    return f'si/{fig}' if fig.startswith('figS') else fig


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figs', nargs='*', help='figures to fetch, e.g. fig1 figS2 (default: all)')
    ap.add_argument('--figures-dir', type=Path, default=Path(__file__).resolve().parent,
                    help='the repository\'s paper/figures folder (default: this script\'s folder)')
    ap.add_argument('--record', default=RECORD, help='Zenodo record id')
    ap.add_argument('--keep-zips', type=Path, help='keep the downloaded zips in this folder')
    a = ap.parse_args()

    meta = json.load(urllib.request.urlopen(f'https://zenodo.org/api/records/{a.record}'))
    zips = {f['key'][len('figure_data_'):-len('.zip')]: f for f in meta['files']
            if f['key'].startswith('figure_data_') and f['key'].endswith('.zip')}
    figs = a.figs or sorted(zips)
    unknown = sorted(set(figs) - set(zips))
    if unknown:
        raise SystemExit(f'not in record {a.record}: {unknown}; available: {sorted(zips)}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fig in figs:
            f = zips[fig]
            dst = a.figures_dir / target(fig)
            if not dst.is_dir():
                raise SystemExit(f'{dst} does not exist: is --figures-dir the repository\'s paper/figures?')
            zp = tmp / f['key']
            urllib.request.urlretrieve(f['links']['self'], zp)
            want = f['checksum'].split(':', 1)[1]
            if md5(zp) != want:
                raise SystemExit(f'{f["key"]}: md5 {md5(zp)} != Zenodo {want}')
            n = 0
            with zipfile.ZipFile(zp) as z:
                for m in z.infolist():
                    if m.is_dir():
                        continue
                    parts = Path(m.filename).parts
                    if parts[:2] != ('figures', fig) or '..' in parts:
                        raise SystemExit(f'{f["key"]}: unexpected member {m.filename}')
                    out = dst.joinpath(*parts[2:])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(out, 'wb') as o:
                        shutil.copyfileobj(src, o)
                    n += 1
            if a.keep_zips:
                a.keep_zips.mkdir(parents=True, exist_ok=True)
                shutil.move(str(zp), a.keep_zips / f['key'])
            print(f'{fig:7s} {n:4d} files -> {dst}')


if __name__ == '__main__':
    main()
