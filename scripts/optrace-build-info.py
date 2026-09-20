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
import sys

SOURCE_SUFFIXES = {'.c','.cpp','.cc','.cxx','.h','.hpp','.inc','.rs','.scala',
                   '.sbt','.py','.cmake','.sh','.toml','.json','.bsv','.v','.sv'}
ROOT_FILES = {'CMakeLists.txt','Makefile','Cargo.lock','Cargo.toml','build.sbt'}
PREFIXES = {
    'llama.cpp-gemmini': ('src/','common/','ggml/','tools/main/','cmake/','scripts/'),
    'IM2P.sim': ('src/','sim/','frontend/','config/','scripts/','fpga/gemmini_hp1/host/'),
    'headers': ('include/','quants/'),
}
EXCLUDED_PARTS = {'node_modules','__pycache__','target','models','.cache','.venv','.git'}

def identity(root: Path, name: str) -> tuple[str,str,dict[str,str]]:
    head = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    paths = subprocess.check_output(['git','-C',str(root),'ls-files','-co','--exclude-standard','-z'])
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
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('llama','sim','headers','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--bits',type=int,choices=(4,8),required=True)
    parser.add_argument('--dim',type=int,choices=(16,32,64),required=True)
    parser.add_argument('--mode',choices=('FULL','STRIPE_PIPELINE'),required=True)
    parser.add_argument('--rmd',choices=('ON','OFF'),required=True)
    parser.add_argument('--rmd-backend',choices=('CPU','WS'),required=True)
    a=parser.parse_args()
    sys.path.insert(0, str(a.sim.resolve()))
    from scripts.gemmini_replay_contract import runtime_binding
    hardware, runtime = runtime_binding(a.runtime_manifest.resolve(), f'a{a.bits}w{a.bits}-d{a.dim}-hp1')
    commits={};digests={};inputs={}
    for name,root in (('llama.cpp-gemmini',a.llama),('IM2P.sim',a.sim),('headers',a.headers)):
        commits[name],digests[name],inputs[name]=identity(root.resolve(),name)
    values={'source_commits':commits,'source_worktree_sha256':digests,
            'profile':f'a{a.bits}w{a.bits}-d{a.dim}-hp1',
            'mode':a.mode,'rmd':a.rmd,'rmd_backend':a.rmd_backend,
            'hardware_contract': hardware, 'runtime_artifact': runtime,
            'input_files_sha256':inputs}
    lines=['#pragma once', '#include <gemmini/optrace.hpp>',
           'namespace ggml::gemmini::optrace {',
           'inline void fill_compiled_optrace_info(RunInfo &r) {',
           f'  r.activation_bits = r.weight_bits = {a.bits}; r.dim = {a.dim};',
           f'  r.profile = {json.dumps(values["profile"])};',
           f'  r.hardware_contract_sha256 = {json.dumps(hardware["sha256"])};',
           f'  r.runtime_manifest_sha256 = {json.dumps(runtime["manifest_sha256"])};',
           '  r.backend = "IM2P_SIM/GEMMINI_HP1";',
           f'  r.mode = "{a.mode};RMD={a.rmd};RMD_BACKEND={a.rmd_backend}";',
           f'  r.residual_enabled = {"true" if a.rmd=="ON" and a.rmd_backend=="WS" else "false"};']
    for field,entries in (('source_commits',commits),('source_worktree_sha256',digests)):
        for key,value in sorted(entries.items()):
            lines.append(f'  r.{field}[{json.dumps(key)}] = {json.dumps(value)};')
    lines.extend(['}', '} // namespace ggml::gemmini::optrace',''])
    write_changed(a.out/'source-identities.json',json.dumps(values,indent=2,sort_keys=True)+'\n')
    write_changed(a.out/'optrace-build-config.hpp','\n'.join(lines))

if __name__=='__main__':
    main()
