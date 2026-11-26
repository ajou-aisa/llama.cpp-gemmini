#!/usr/bin/env python3
"""Generate producer source identities without tensor/model/build contents.

The manifest is emitted independently of the trace writer. Re-run at each build
so a dirty source edit cannot keep a stale compile-time provenance header.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_SUFFIXES = {'.c','.cpp','.cc','.cxx','.h','.hpp','.inc','.rs','.scala',
                   '.sbt','.py','.cmake','.sh','.toml','.json','.bsv','.v','.sv'}
ROOT_FILES = {'CMakeLists.txt','Makefile','Cargo.lock','Cargo.toml','build.sbt'}
PREFIXES = {
    'llama.cpp-gemmini': ('src/','common/','ggml/','tools/main/','cmake/','scripts/'),
    'headers': ('api/','legacy/','matmul/','quants/','residual/'),
}
EXCLUDED_PARTS = {'node_modules','__pycache__','target','models','.cache','.venv','.git'}

def identity(root: Path, name: str) -> tuple[str,str,dict[str,str]]:
    commit = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'], capture_output=True, text=True, check=False)
    head = commit.stdout.strip() if commit.returncode == 0 else ''
    listed = subprocess.run(['git','-C',str(root),'ls-files','-co','--exclude-standard','-z'], capture_output=True, check=False)
    if listed.returncode == 0:
        paths = listed.stdout
    else:
        candidates = [p for prefix in PREFIXES[name] for p in (root / prefix).rglob('*')]
        candidates.extend(p for p in root.iterdir() if p.is_file())
        paths = b'\0'.join(str(p.relative_to(root)).encode() for p in candidates if p.is_file())
    hashes: dict[str,str] = {}
    for raw in sorted(set(paths.split(b'\0'))):
        if not raw:
            continue
        relative = raw.decode('utf-8')
        p = Path(relative)
        if any(part in EXCLUDED_PARTS or part.startswith('build') for part in p.parts[:-1]):
            continue
        if not (relative in ROOT_FILES or relative.startswith(PREFIXES[name]) or
                (name=='headers' and len(p.parts)==1)):
            continue
        if p.suffix not in SOURCE_SUFFIXES and p.name not in ROOT_FILES:
            continue
        source = root/p
        if not source.is_file() or source.is_symlink():
            continue
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError(f'no source inputs for {name}')
    encoded = json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()
    return head, hashlib.sha256(encoded).hexdigest(), hashes


def write_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text()==content:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(content)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('llama', 'headers', 'out', 'cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--bits', type=int, required=True)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--mode', choices=('FULL', 'STRIPE_PIPELINE'), required=True)
    parser.add_argument('--rmd', choices=('ON', 'OFF'), required=True)
    a = parser.parse_args()
    commits, digests, sources = {}, {}, {}
    for name, root in (('llama.cpp-gemmini', a.llama), ('headers', a.headers)):
        head, digest, hashes = identity(root.resolve(), name)
        commits[name], digests[name], sources[name] = head, digest, hashes
    configuration = {}
    for line in a.cache.read_text().splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=', 1)[0]:
            continue
        key, value = line.split('=', 1)
        key = key.split(':', 1)[0]
        if key.startswith(('GGML_', 'NPU_', 'LOG_', 'CYCLE_DETAIL')):
            configuration[key] = value
    encoded = json.dumps(configuration, sort_keys=True, separators=(',', ':')).encode()
    hardware = {key: value for key, value in configuration.items() if key.startswith('NPU_')}
    values = dict(profile=f'a{a.bits}w{a.bits}-d{a.dim}-hp1', activation_bits=a.bits,
                  weight_bits=a.bits, dim=a.dim, backend=a.backend, mode=a.mode,
                  residual_enabled=a.rmd == 'ON', source_commits=commits,
                  source_worktree_sha256=digests,
                  hardware_contract_sha256=hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest(),
                  runtime_manifest_sha256=hashlib.sha256(encoded).hexdigest())
    lines = ['#pragma once', '#include <npu/optrace.hpp>',
             'namespace ggml::npu::optrace {',
             'inline void fill_compiled_optrace_info(RunInfo &r) {']
    for key, value in values.items():
        if not isinstance(value, dict):
            lines.append(f'  r.{key} = {json.dumps(value)};')
    for field, entries in (('source_commits', commits), ('source_worktree_sha256', digests)):
        for key, value in sorted(entries.items()):
            lines.append(f'  r.{field}[{json.dumps(key)}] = {json.dumps(value)};')
    lines.extend(['}', '} // namespace ggml::npu::optrace', ''])
    manifest = {**values, 'configuration': configuration, 'source_files': sources,
                'identity_kind': 'native-build-configuration; empty commit means VCS unavailable'}
    write_changed(a.out / 'source-identities.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    write_changed(a.out / 'optrace-build-config.hpp', '\n'.join(lines))

if __name__ == '__main__':
    main()
