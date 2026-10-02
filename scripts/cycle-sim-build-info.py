#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: uv run scripts/cycle-sim-build-info.py --help
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def write_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Bind CPU-functional build to canonical target hardware.')
    parser.add_argument('--sim', type=Path, required=True)
    parser.add_argument('--llama', type=Path, required=True)
    parser.add_argument('--bits', type=int, choices=(4, 8), required=True)
    parser.add_argument('--dim', type=int, choices=(16, 32, 64), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sim.resolve()))
    from scripts.gemmini_replay_contract import canonical_json, hardware_contract, validate_contract

    profile = f'a{args.bits}w{args.bits}-d{args.dim}-hp1'
    contract = hardware_contract(profile, args.sim.resolve())
    validate_contract(contract, profile)
    facts = contract['facts']
    if not isinstance(facts, dict) or not isinstance(facts.get('memory'), dict):
        parser.error('resolved target memory contract unavailable')
    memory = facts['memory']
    if not isinstance(memory, dict):
        parser.error('resolved target memory must be an object')
    fields = {'ACCUMULATOR_ROWS': memory['accumulator_rows'], 'BANK_COUNT': memory['bank_count'],
              'BANK_ROWS': memory['bank_rows'], 'PARTIAL_BITS': facts['accumulator_bits'],
              'HARDWARE_CONTRACT_SHA256': contract['sha256']}
    target = ''.join(f'set(IM2P_CYCLE_SIM_{name} {json.dumps(value)})\n' for name, value in fields.items())
    source_hashes = contract['source_sha256']
    if not isinstance(source_hashes, dict):
        parser.error('resolved hardware source closure unavailable')
    inputs = [args.sim.resolve() / name for name in source_hashes]
    inputs.extend((Path(__file__).resolve(), args.sim.resolve() / 'scripts/gemmini_replay_contract.py',
                   args.sim.resolve() / 'scripts/gemmini_resolve_profile.py'))
    target += 'set(IM2P_CYCLE_SIM_CONFIGURE_INPUTS\n' + ''.join(
        f'  {json.dumps(str(path))}\n' for path in sorted(inputs)) + ')\n'
    header = '\n'.join([
        '#pragma once', '#include <gemmini/cycle_sim_log.hpp>',
        'namespace ggml::gemmini::cycle_sim {',
        'inline void fill_compiled_cycle_sim_info(RunInfo &info) {',
        f'    info.profile = {json.dumps(profile)};',
        f'    info.activation_bits = info.weight_bits = {args.bits}; info.dim = {args.dim};',
        f'    info.hardware_contract_sha256 = {json.dumps(contract["sha256"])};',
        f'    info.hardware_contract_json = {json.dumps(canonical_json(contract))};',
        '}', '}', '',
    ])
    write_changed(args.out / 'cycle-sim-target.cmake', target)
    write_changed(args.out / 'cycle-sim-build-config.hpp', header)
    write_changed(args.out / 'cycle-sim-hardware-contract.json', canonical_json(contract) + '\n')


if __name__ == '__main__':
    main()
