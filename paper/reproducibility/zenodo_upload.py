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

"""Stage and upload the paper's Zenodo records as PRIVATE DRAFTS. Never publishes.

    plan    list every record's files, sizes and problems (missing sources, >100 files, >50 GB)
    stage   build the zips / code tarball into --staging (skips ones already built unless --rebuild)
    upload  create one draft per record (or reuse the one in --state), reserve its DOI, set metadata,
            upload the staged files (same name + md5 already on the draft = skipped, differs = replaced)

Everything comes from the YAML spec (records, items, metadata); paths never live here. Paths in the spec and in
plot_time_inputs.tsv may contain environment variables ($EXPERT_ROOT, $DL_ROOT, ...; see paper/slurm/paths.sh),
expanded when read. The token is read
from the environment variable named by --token-env and needs only the deposit:write scope. There is no
publish call anywhere in this file: publishing stays a manual click on zenodo.org.
Related identifiers: `defaults.related_identifiers` (Zenodo format: identifier, relation, resource_type) go on every
record; a record's `related: [{record: <key>, relation}]` links to a sibling record by its reserved DOI, resolved
from the state file after all drafts of the run exist (a sibling without a draft yet is left out until a rerun).
A record with `opt_in: true` (the code tarball, for the final publication only) is skipped unless --opt-in.

Item kinds in the spec:
    file     {path, name?}                         uploaded as is
    zip      {name, members: [{path, arc?, include?, exclude?}]}
             a member path may be a file or a directory; include/exclude are fnmatch globs on the path
             relative to the member root
    figures  {tsv, held: [{figure, pattern}], prefix?}
             one zip per figure from plot_time_inputs.tsv (arc = figures/<fig>/<notebook_path>); rows
             whose notebook_path matches a held pattern for that figure are left out
    git      {repo, ref, name, exclude: [pathspecs]}   git archive of one commit as .tar.gz
"""
import argparse, fnmatch, hashlib, json, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

MAX_FILES, MAX_BYTES = 100, 50e9
STORED = ('.npz', '.npy.gz', '.gz', '.zip', '.parquet', '.h5ad', '.h5', '.png', '.pdf', '.jpg', '.tif', '.tiff')
EPOCH = (1980, 1, 1, 0, 0, 0)


def gb(n):
    return f'{n / 1e9:.2f} GB' if n >= 1e8 else f'{n / 1e6:.1f} MB'


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# -- what goes into each record ---------------------------------------------------------------------

def _walk(m):
    """(source, arcname) pairs of one zip member, filtered by its include/exclude globs."""
    root = Path(os.path.expandvars(m['path']))
    arc = m.get('arc', root.name)
    if root.is_file():
        return [(root, arc)]
    inc, exc = m.get('include'), m.get('exclude', [])
    # include globs are expanded from the root (a matched directory brings its whole subtree), so a
    # narrow include never walks a large tree; without include the whole root is walked
    if inc:
        hits = set()
        for g in inc:
            for p in root.glob(g):
                hits.update(p.rglob('*') if p.is_dir() else [p])
    else:
        hits = root.rglob('*')
    out = []
    for p in sorted(hits):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, g) for g in exc):
            continue
        out.append((p, f'{arc}/{rel}'))
    return out


def _figure_zips(item):
    t = pd.read_csv(item['tsv'], sep='\t')
    held = pd.Series(False, index=t.index)
    for h in item.get('held', []):
        held |= (t['figure'] == h['figure']) & t['notebook_path'].str.contains(h['pattern'], regex=True)
    out = []
    for fig, g in t[~held].groupby('figure', sort=True):
        members = [(Path(os.path.expandvars(r.real_path)), f'figures/{fig}/{r.notebook_path}') for r in g.itertuples()]
        out.append({'name': f"{item.get('prefix', 'figure_data_')}{fig}.zip", 'members': members})
    return out, int(held.sum()), int(t.loc[held, 'bytes'].sum())


def expand(rec, staging):
    """Resolve one record into upload units: {name, src (path to upload), build (None | zip | git spec)}."""
    units, notes = [], []
    sdir = Path(staging) / rec['key']
    for it in rec['items']:
        kind = it['kind']
        if kind == 'file':
            p = Path(os.path.expandvars(it['path']))
            units.append({'name': it.get('name', p.name), 'src': p, 'build': None,
                          'bytes': p.stat().st_size if p.exists() else None})
        elif kind == 'zip':
            members = [x for m in it['members'] for x in _walk(m)]
            units.append({'name': it['name'], 'src': sdir / it['name'], 'build': ('zip', members),
                          'bytes': sum(p.stat().st_size for p, _ in members)})
        elif kind == 'figures':
            zips, n_held, b_held = _figure_zips(it)
            for z in zips:
                units.append({'name': z['name'], 'src': sdir / z['name'], 'build': ('zip', z['members']),
                              'bytes': sum(p.stat().st_size for p, _ in z['members'] if p.exists())})
            notes.append(f'{n_held} figure-data files ({gb(b_held)}) held back by the spec')
        elif kind == 'git':
            units.append({'name': it['name'], 'src': sdir / it['name'], 'build': ('git', it), 'bytes': None})
        else:
            raise ValueError(f'unknown item kind {kind!r} in record {rec["key"]}')
    # every record carries a CONTENTS.tsv listing each archive member
    units.append({'name': 'CONTENTS.tsv', 'src': sdir / 'CONTENTS.tsv', 'build': ('contents', None), 'bytes': 0})
    return units, notes


# -- stage ------------------------------------------------------------------------------------------

def zip_fingerprint(members):
    """Member list + sizes + mtimes: a staged zip whose sources changed since it was built is rebuilt."""
    h = hashlib.md5()
    for p, arc in sorted(members, key=lambda x: x[1]):
        st = p.stat()
        h.update(f'{arc}\t{st.st_size}\t{int(st.st_mtime)}\n'.encode())
    return h.hexdigest()


def sidecar(path):
    return path.with_name(path.name + '.members')


def _read(path):
    return path.read_text() if path.exists() else None


def build_zip(dst, members):
    missing = [str(p) for p, _ in members if not p.exists()]
    if missing:
        raise FileNotFoundError(f'{dst.name}: {len(missing)} missing sources, first {missing[0]}')
    tmp = dst.with_suffix(dst.suffix + '.part')
    with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
        for p, arc in sorted(members, key=lambda x: x[1]):
            # fixed timestamps and order so a rebuild of unchanged sources has the same md5
            zi = zipfile.ZipInfo(arc, EPOCH)
            zi.external_attr = 0o644 << 16
            zi.compress_type = zipfile.ZIP_STORED if p.name.lower().endswith(STORED) else zipfile.ZIP_DEFLATED
            with open(p, 'rb') as fi, z.open(zi, 'w', force_zip64=True) as fo:
                for b in iter(lambda: fi.read(1 << 24), b''):
                    fo.write(b)
    tmp.rename(dst)


def build_git(dst, spec):
    sha = subprocess.check_output(['git', '-C', spec['repo'], 'rev-parse', spec['ref']], text=True).strip()
    prefix = spec.get('prefix', dst.name.removesuffix('.tar.gz')) + '/'
    paths = ['.'] + [f':(exclude){x}' for x in spec.get('exclude', [])]
    tmp = dst.with_suffix(dst.suffix + '.part')
    subprocess.run(['git', '-C', spec['repo'], 'archive', '--format=tar.gz', f'--prefix={prefix}',
                    '-o', str(tmp.resolve()), sha, '--'] + paths, check=True)
    tmp.rename(dst)
    return sha


def stage(rec, units, rebuild):
    sdir = Path(units[-1]['src']).parent
    sdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for u in units:
        b = u['build']
        if b is None or b[0] == 'contents':
            if b is None:
                rows.append((u['name'], '', u['bytes']))
            continue
        if b[0] == 'zip':
            rows += [(u['name'], arc, p.stat().st_size) for p, arc in b[1]]
        stale = b[0] == 'zip' and u['src'].exists() and zip_fingerprint(b[1]) != _read(sidecar(u['src']))
        if u['src'].exists() and not rebuild and not stale:
            print(f'  [keep] {u["name"]} ({gb(u["src"].stat().st_size)})')
            if b[0] == 'git':
                rows.append((u['name'], f'git {b[1]["ref"]}', u['src'].stat().st_size))
            continue
        if b[0] == 'git' and shutil.which('git') is None:
            # compute nodes have no git; the tarball takes seconds, so it is built on a login node
            print(f'  [skip] {u["name"]}: no git on this node -- run `zenodo_upload.py stage --records '
                  f'{rec["key"]} ...` on a login node')
            continue
        print(f'  [build] {u["name"]}{" (members changed)" if stale else ""} ...', flush=True)
        if b[0] == 'zip':
            build_zip(u['src'], b[1])
            sidecar(u['src']).write_text(zip_fingerprint(b[1]))
        else:
            sha = build_git(u['src'], b[1])
            rows.append((u['name'], f'git {b[1]["ref"]} = {sha}', u['src'].stat().st_size))
        print(f'  [built] {u["name"]} ({gb(u["src"].stat().st_size)})', flush=True)
    pd.DataFrame(rows, columns=['file', 'member', 'bytes']).to_csv(sdir / 'CONTENTS.tsv', sep='\t', index=False)


# -- plan -------------------------------------------------------------------------------------------

def plan(rec, units, notes):
    ok = True
    total = sum(u['bytes'] or 0 for u in units)
    print(f'\n== {rec["key"]}: {rec["metadata"]["title"]}')
    print(f'   {len(units)} files, ~{gb(total)} before compression')
    for u in units:
        b = u['build']
        what = 'file' if b is None else b[0]
        extra = ''
        if b is not None and b[0] == 'zip':
            miss = sum(not p.exists() for p, _ in b[1])
            extra = f'  {len(b[1])} members' + (f'  ** {miss} MISSING **' if miss else '')
            ok &= not miss
        if b is None and u['bytes'] is None:
            extra = '  ** MISSING **'
            ok = False
        size = gb(u['bytes']) if u['bytes'] else '-'
        print(f'   {what:8s} {u["name"]:55s} {size:>10s}{extra}')
    for n in notes:
        print(f'   note: {n}')
    if len(units) > MAX_FILES:
        print(f'   ** {len(units)} files > Zenodo limit {MAX_FILES} **'); ok = False
    if total > MAX_BYTES:
        print(f'   ** {gb(total)} > Zenodo limit 50 GB **'); ok = False
    return ok


# -- upload (drafts only) ---------------------------------------------------------------------------

class Zenodo:
    def __init__(self, base, token):
        self.base = base.rstrip('/')
        self.s = requests.Session()
        self.s.headers['Authorization'] = f'Bearer {token}'

    def _ok(self, r):
        if r.status_code >= 400:
            raise RuntimeError(f'{r.request.method} {r.url} -> {r.status_code}: {r.text[:500]}')
        return r.json() if r.content else {}

    def draft(self, dep_id=None):
        if dep_id is None:
            return self._ok(self.s.post(f'{self.base}/api/deposit/depositions', json={}))
        r = self.s.get(f'{self.base}/api/deposit/depositions/{dep_id}')
        if r.status_code == 404:
            return None
        return self._ok(r)

    def is_published(self, rec_id):
        return self.s.get(f'{self.base}/api/records/{rec_id}').status_code == 200

    def set_metadata(self, dep_id, md):
        return self._ok(self.s.put(f'{self.base}/api/deposit/depositions/{dep_id}', json={'metadata': md}))

    def files(self, dep_id):
        return self._ok(self.s.get(f'{self.base}/api/deposit/depositions/{dep_id}/files'))

    def delete_file(self, dep_id, file_id):
        self._ok(self.s.delete(f'{self.base}/api/deposit/depositions/{dep_id}/files/{file_id}'))

    def put(self, bucket, name, path):
        with open(path, 'rb') as f:
            return self._ok(self.s.put(f'{bucket}/{name}', data=f))


ORCID = re.compile(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$')


def clean_person(p):
    """Drop an empty ORCID placeholder; Zenodo rejects '' and malformed ids."""
    p = dict(p)
    o = str(p.pop('orcid', '') or '').strip()
    if o:
        if not ORCID.match(o):
            raise ValueError(f'{p["name"]}: ORCID {o!r} is not 0000-0000-0000-000X')
        p['orcid'] = o
    return p


def record_metadata(rec, defaults, state):
    md = {**defaults, **rec['metadata'], 'prereserve_doi': True}
    md['creators'] = [clean_person(p) for p in md.get('creators', [])]
    md['contributors'] = [clean_person(p) for p in md.get('contributors', [])]
    rel = list(md.get('related_identifiers', []))
    for r in rec.get('related', []):
        doi = state.get(r['record'], {}).get('doi')
        if doi:
            rel.append({'identifier': doi, 'relation': r['relation'], 'resource_type': r.get('resource_type', 'dataset')})
        else:
            print(f'  [related] {rec["key"]} {r["relation"]} {r["record"]}: no draft yet, left out until a rerun')
    if rel:
        md['related_identifiers'] = rel
    return md


def upload(rec, units, z, state, defaults):
    key = rec['key']
    st = state.get(key, {})
    d = z.draft(st.get('id'))
    if d is None:
        # the stored draft is gone: published (it then lives under /records) or deleted on the website
        if z.is_published(st['id']):
            raise RuntimeError(f'{key}: record {st["id"]} is PUBLISHED; refusing to touch it. To start a new draft '
                               f'instead, remove "{key}" from the state file.')
        print(f'\n== {key}: stored draft {st["id"]} no longer exists (deleted?) -> creating a new draft')
        st = {}
        d = z.draft()
    elif d.get('submitted') or d.get('state') == 'done':
        raise RuntimeError(f'{key}: deposition {d["id"]} is already PUBLISHED; refusing to touch it. To start a new '
                           f'draft instead, remove "{key}" from the state file.')
    d = z.set_metadata(d['id'], record_metadata(rec, defaults, state))
    doi = (d.get('metadata', {}).get('prereserve_doi') or {}).get('doi')
    st.update(id=d['id'], doi=doi, html=d['links']['html'], bucket=d['links']['bucket'])
    state[key] = st
    print(f'\n== {key}: draft {d["id"]}  reserved DOI {doi}  {d["links"]["html"]}', flush=True)
    remote = {f['filename']: f for f in z.files(d['id'])}
    for u in units:
        p = Path(u['src'])
        if not p.exists():
            raise FileNotFoundError(f'{key}: {p} not staged; run `stage` first')
        r = remote.pop(u['name'], None)
        if r is not None:
            if r.get('filesize') == p.stat().st_size and r.get('checksum') == md5(p):
                print(f'  [same] {u["name"]}')
                continue
            z.delete_file(d['id'], r['id'])
            print(f'  [replace] {u["name"]}')
        print(f'  [put] {u["name"]} ({gb(p.stat().st_size)}) ...', flush=True)
        z.put(st['bucket'], u['name'], p)
    for name in remote:
        print(f'  [extra on draft, left alone] {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('plan', 'stage', 'upload'))
    ap.add_argument('--spec', required=True, help='records YAML')
    ap.add_argument('--staging', required=True, help='where zips / tarball are built')
    ap.add_argument('--state', required=True, help='JSON with each record\'s draft id + reserved DOI')
    ap.add_argument('--records', nargs='*', help='subset of record keys (default all)')
    ap.add_argument('--rebuild', action='store_true', help='stage: rebuild archives that already exist')
    ap.add_argument('--sandbox', action='store_true', help='upload to sandbox.zenodo.org (test drafts)')
    ap.add_argument('--opt-in', action='store_true', help='also handle records marked opt_in: true')
    ap.add_argument('--token-env', default='ZENODO_TOKEN', help='env var holding the deposit:write token')
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    spec = yaml.safe_load(open(a.spec))
    recs = [r for r in spec['records'] if not a.records or r['key'] in a.records]
    if a.records and len(recs) != len(a.records):
        sys.exit(f'unknown record(s): {set(a.records) - {r["key"] for r in recs}}')
    skipped = [r['key'] for r in recs if r.get('opt_in') and not a.opt_in]
    if skipped:
        print(f'skipping opt-in record(s) {skipped} (pass --opt-in to include)')
        recs = [r for r in recs if r['key'] not in skipped]

    if a.cmd == 'upload':
        token = os.environ.get(a.token_env)
        if not token:
            sys.exit(f'set ${a.token_env} (Zenodo token, scope deposit:write)')
        z = Zenodo('https://sandbox.zenodo.org' if a.sandbox else 'https://zenodo.org', token)
        state_path = Path(a.state)
        if a.sandbox:
            state_path = state_path.with_name(state_path.stem + '.sandbox' + state_path.suffix)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}

    ok = True
    for rec in recs:
        units, notes = expand(rec, a.staging)
        if a.cmd == 'plan':
            ok &= plan(rec, units, notes)
        elif a.cmd == 'stage':
            print(f'\n== {rec["key"]}')
            stage(rec, units, a.rebuild)
        else:
            try:
                upload(rec, units, z, state, spec.get('defaults', {}))
            finally:
                state_path.write_text(json.dumps(state, indent=2))   # keep draft ids even if a PUT fails
    if a.cmd == 'upload':
        # second pass: every sibling now has a reserved DOI, so the record-to-record links resolve
        for rec in recs:
            if rec.get('related'):
                md = record_metadata(rec, spec.get('defaults', {}), state)
                d = z.draft(state[rec['key']]['id'])
                if d is None or d.get('submitted'):
                    # published meanwhile: the deposit API cannot edit it without a publish action, which this
                    # script never takes -> add the links in the web UI (Edit > Related works > Save + Publish)
                    print(f'== {rec["key"]}: already published, related identifiers NOT set; add on zenodo.org: ' +
                          ', '.join(f'{r["relation"]} {r["identifier"]}' for r in md.get('related_identifiers', [])))
                    continue
                z.set_metadata(state[rec['key']]['id'], md)
                print(f'== {rec["key"]}: related identifiers ' +
                      ', '.join(f'{r["relation"]} {r["identifier"]}' for r in md.get('related_identifiers', [])))
    if a.cmd == 'plan':
        print('\nPLAN OK' if ok else '\nPLAN HAS PROBLEMS (see ** above)')
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
