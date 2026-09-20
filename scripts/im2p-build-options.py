#!/usr/bin/env python3
"""Resolve build-script options before provisioning. No build or device access."""
import json
import os
from pathlib import Path
import re
import shlex
import sys

FPGA = {
    'GGML_GEMMINI': 'ON', 'GGML_GEMMINI_OPTION': 'WS',
    'GGML_GEMMINI_COMPUTE_TYPE': 'INT', 'GGML_GEMMINI_ACTIVATION_QUANT': 'EXSIA',
    'GGML_GEMMINI_ACTIVATION_BITS': '8', 'GGML_GEMMINI_WEIGHT_BITS': '8',
    'GGML_GEMMINI_DIM': '16', 'GGML_GEMMINI_BLOCK_SIZE': '32',
    'GGML_GEMMINI_ENABLE_RMD': 'OFF', 'GGML_GEMMINI_DEQUANT_FP_TEST': 'OFF',
}


class BuildConfigurationError(ValueError):
    pass


def load_hp1_profile(path, require_rmd: bool = False):
    try:
        profile = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BuildConfigurationError(f'Cannot read GEMMINI_HP1 resolved profile {path!r}: {error}') from error
    if not isinstance(profile, dict):
        raise BuildConfigurationError('GEMMINI_HP1 resolved profile must be a JSON object')
    integer_fields = ('schema_version', 'activation_bits', 'weight_bits', 'dim',
                      'accumulator_bits', 'block_size')
    if any(type(profile.get(name)) is not int for name in integer_fields):
        raise BuildConfigurationError('GEMMINI_HP1 resolved profile integer fields are missing or invalid')
    bits, weight_bits, dim = profile['activation_bits'], profile['weight_bits'], profile['dim']
    packing = 'signed-int4-low-nibble-first' if bits == 4 else 'signed-int8'
    expected = {
        'schema_version': 1,
        'profile': f'a{bits}w{weight_bits}-d{dim}-hp1',
        'implementation': 'gemmini-hp1',
        'accumulator_bits': 32,
        'block_size': 32,
        'scu': 'hp1-left-shift',
        'packing': packing,
        'numerical_revision': 'hp1-fragment-sat32-v1',
    }
    if bits not in (4, 8) or weight_bits != bits or dim not in (16, 32, 64):
        raise BuildConfigurationError(f'Unsupported GEMMINI_HP1 profile A{bits}/W{weight_bits}/DIM{dim}')
    for name, value in expected.items():
        if profile.get(name) != value:
            raise BuildConfigurationError(f'GEMMINI_HP1 resolved profile requires {name}={value!r}')
    memory = profile.get('memory')
    memory_integer_fields = (
        'bank_count', 'bank_rows', 'accumulator_rows', 'scratchpad_row_bytes',
        'accumulator_row_bytes', 'scratchpad_total_bytes', 'accumulator_total_bytes',
        'ws_scratchpad_rows_per_buffer', 'ws_accumulator_rows_per_buffer')
    if not isinstance(memory, dict) or any(type(memory.get(name)) is not int for name in memory_integer_fields):
        raise BuildConfigurationError('GEMMINI_HP1 resolved profile memory fields are missing or invalid')
    memory_valid = (
        memory['bank_count'] == 4 and memory.get('ws_double_buffered') is True and
        memory['scratchpad_row_bytes'] == dim * bits // 8 and
        memory['accumulator_row_bytes'] == dim * 4 and
        memory['bank_count'] * memory['bank_rows'] * memory['scratchpad_row_bytes'] ==
        memory['scratchpad_total_bytes'] and
        memory['accumulator_rows'] * memory['accumulator_row_bytes'] ==
        memory['accumulator_total_bytes'] and
        memory['ws_scratchpad_rows_per_buffer'] == memory['bank_count'] * memory['bank_rows'] // 2 and
        memory['ws_accumulator_rows_per_buffer'] == memory['accumulator_rows'] // 2)
    if not memory_valid:
        raise BuildConfigurationError('GEMMINI_HP1 resolved profile memory contract is inconsistent')
    contract_hash = profile.get('host_contract_sha256')
    if not profile.get('host_contract_source') or not isinstance(contract_hash, str) or \
            len(contract_hash) != 64 or any(character not in '0123456789abcdef' for character in contract_hash):
        raise BuildConfigurationError('GEMMINI_HP1 resolved profile host contract identity is invalid')
    if require_rmd and (profile.get('rmd_raw') is not True or
                        profile.get('rmd_numerical_revision') != 'rmd-raw-k32-cpu-compose-v1'):
        raise BuildConfigurationError('GEMMINI_HP1 RMD ON requires the RMD_RAW manifest contract')
    return profile


def apply_hp1_profile(effective, origin):
    manifest = effective.get('IM2P_GEMMINI_RESOLVED_PROFILE')
    if not manifest:
        raise BuildConfigurationError('GEMMINI_HP1 requires IM2P_GEMMINI_RESOLVED_PROFILE')
    profile = load_hp1_profile(manifest, effective.get('GGML_GEMMINI_ENABLE_RMD') == 'ON')
    resolved = {
        'GGML_GEMMINI_ACTIVATION_BITS': str(profile['activation_bits']),
        'GGML_GEMMINI_WEIGHT_BITS': str(profile['weight_bits']),
        'GGML_GEMMINI_DIM': str(profile['dim']),
        'GGML_GEMMINI_BLOCK_SIZE': str(profile['block_size']),
    }
    for name, required in resolved.items():
        if origin.get(name) in ('environment', 'command-line') and effective.get(name) != required:
            raise BuildConfigurationError(
                f'GEMMINI_HP1 resolved profile requires {name}={required}; '
                f'got {effective.get(name)!r} from {origin[name]}')
        effective[name] = required
        origin[name] = 'resolved-profile'
    return resolved


def configurable(name):
    return name.startswith(('GGML_', 'IM2P_', 'LOG_', 'CYCLE_', 'CMAKE_', 'LLAMA_')) or name in ('BUILD_SHARED_LIBS', 'OpenMP_ROOT')


def normalize(name, value):
    if name in ('GGML_GEMMINI', 'GGML_GEMMINI_ENABLE_RMD', 'GGML_GEMMINI_DEQUANT_FP_TEST'):
        if value.upper() in ('ON', 'YES', 'TRUE', '1'):
            return 'ON'
        if value.upper() in ('OFF', 'NO', 'FALSE', '0'):
            return 'OFF'
    return value


def resolve(build_dir, platform, defaults, args, environment):
    cli, passthrough, dry_run = {}, [], False
    for arg in args:
        if arg == '--dry-run':
            dry_run = True
        elif arg.startswith('-D'):
            match = re.fullmatch(r'-D([A-Za-z_][A-Za-z0-9_]*)(?::(BOOL|FILEPATH|PATH|STRING|INTERNAL))?=(.*)', arg, re.S)
            if not match:
                raise BuildConfigurationError(f'Unsupported -D syntax: {arg}; use -DNAME[:TYPE]=value')
            cli[match[1]] = match[3]
            passthrough.append(arg)
        elif arg in ('-Wdev', '-Wno-dev', '-Werror=dev', '-Wno-error=dev', '--warn-uninitialized', '--debug-find'):
            passthrough.append(arg)
        else:
            raise BuildConfigurationError(f'Unsupported script configure argument: {arg}; use direct CMake for presets, toolchains, -C/-U/-S/-B/-G')
    if platform != 'build-riscv.sh' and ('CMAKE_TOOLCHAIN_FILE' in cli or environment.get('CMAKE_TOOLCHAIN_FILE')):
        raise BuildConfigurationError('Native build scripts do not provision cross toolchains; use direct CMake with explicit target artifacts')

    env = {k: v for k, v in environment.items() if configurable(k) and not k.endswith('_DEFAULT')}
    for key, value in environment.items():
        if key.endswith('_DEFAULT') and configurable(key[:-8]):
            env.setdefault(key[:-8], value)
    # The alias is resolved at its own precedence level; same-level disagreement is an error.
    for level in (defaults, env, cli):
        if 'IM2P_DIM' in level:
            if 'GGML_GEMMINI_DIM' in level and level['IM2P_DIM'] != level['GGML_GEMMINI_DIM']:
                raise BuildConfigurationError('IM2P_DIM and GGML_GEMMINI_DIM must match when set at the same precedence level')
            level['GGML_GEMMINI_DIM'] = level.pop('IM2P_DIM')
    effective = dict(defaults)
    origin = {key: 'default' for key in defaults}
    for label, level in (('environment', env), ('command-line', cli)):
        effective.update(level)
        origin.update({key: label for key in level})
    if origin.get('GGML_CPU_CYCLE_LOG', 'default') == 'default':
        effective['GGML_CPU_CYCLE_LOG'] = effective.get('LOG_CYCLE', '0')
        origin['GGML_CPU_CYCLE_LOG'] = 'derived-default:LOG_CYCLE'
    effective = {key: normalize(key, value) for key, value in effective.items()}
    cycle_sim = effective.get('CYCLE_SIM', '0')
    if cycle_sim not in ('0', '1'):
        raise BuildConfigurationError('CYCLE_SIM must be 0 or 1')
    backend = effective.get('GGML_GEMMINI_EXECUTION_BACKEND', 'HARDWARE')
    if cycle_sim == '1' and backend == 'FPGA_UART':
        raise BuildConfigurationError('CYCLE_SIM cannot use FPGA_UART')
    if backend not in ('HARDWARE', 'IM2P_SIM', 'FPGA_UART'):
        raise BuildConfigurationError(f'Unknown GGML_GEMMINI_EXECUTION_BACKEND={backend}')
    if platform == 'build-riscv.sh' and backend != 'HARDWARE' and cycle_sim != '1':
        raise BuildConfigurationError('build-riscv.sh is the HARDWARE lane; FPGA_UART uses the native x86/ARM64 script')
    implementation = effective.get('IM2P_SIM_IMPLEMENTATION', 'LEGACY_BSV')
    if implementation not in ('GEMMINI_HP1', 'LEGACY_BSV'):
        raise BuildConfigurationError(f'Unknown IM2P_SIM_IMPLEMENTATION={implementation}')
    if backend != 'IM2P_SIM':
        if origin.get('IM2P_SIM_IMPLEMENTATION') in ('environment', 'command-line'):
            raise BuildConfigurationError(
                'IM2P_SIM_IMPLEMENTATION is valid only with '
                'GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM')
        effective['IM2P_SIM_IMPLEMENTATION'] = ''
        origin['IM2P_SIM_IMPLEMENTATION'] = 'cleared:non-IM2P_SIM'
    else:
        effective['IM2P_SIM_IMPLEMENTATION'] = implementation
        if 'IM2P_SIM_IMPLEMENTATION' not in origin:
            origin['IM2P_SIM_IMPLEMENTATION'] = 'IM2P_SIM default'
        if effective.get('IM2P_GEMMINI_RESOLVED_PROFILE'):
            if origin.get('IM2P_GEMMINI_RESOLVED_PROFILE') in ('environment', 'command-line'):
                raise BuildConfigurationError(
                    'IM2P_GEMMINI_RESOLVED_PROFILE is valid only with '
                    'GGML_GEMMINI_EXECUTION_BACKEND=FPGA_UART')
            effective.pop('IM2P_GEMMINI_RESOLVED_PROFILE', None)
            origin.pop('IM2P_GEMMINI_RESOLVED_PROFILE', None)
    fpga_arch = effective.get('IM2P_FPGA_ARCH', '')
    if fpga_arch and backend != 'FPGA_UART':
        raise BuildConfigurationError('IM2P_FPGA_ARCH is valid only with GGML_GEMMINI_EXECUTION_BACKEND=FPGA_UART')
    if fpga_arch not in ('', 'BSV_IFR4', 'GEMMINI_HP1'):
        raise BuildConfigurationError(f'Unknown IM2P_FPGA_ARCH={fpga_arch}')
    if backend == 'FPGA_UART':
        if effective.get('GGML_GEMMINI_FPGA_SIM_MANIFEST'):
            raise BuildConfigurationError('GGML_GEMMINI_FPGA_SIM_MANIFEST is invalid for FPGA_UART; physical external executor uses no simulator archive')
        required_profile = dict(FPGA)
        profile_label = 'FPGA_UART'
        if fpga_arch == 'GEMMINI_HP1':
            profile_label = 'GEMMINI_HP1'
            resolved = apply_hp1_profile(effective, origin)
            required_profile.update(resolved)
        for name, required in required_profile.items():
            if name == 'GGML_GEMMINI_ACTIVATION_QUANT' and origin.get(name, 'default') != 'default':
                effective[name] = effective[name].upper()
                if effective[name] not in ('EXSIA', 'TENSOR', 'TOKEN', 'BLOCK', 'STRIPE'):
                    raise BuildConfigurationError(f'{profile_label} activation must be EXSIA, TENSOR, TOKEN, BLOCK, or STRIPE')
                continue
            if name == 'GGML_GEMMINI_ENABLE_RMD' and origin.get(name, 'default') != 'default':
                if effective.get(name) not in ('ON', 'OFF'):
                    raise BuildConfigurationError('GGML_GEMMINI_ENABLE_RMD must be ON or OFF')
                continue
            if origin.get(name, 'default') != 'default' and effective.get(name) != required:
                raise BuildConfigurationError(f'{profile_label} requires {name}={required}; got {effective.get(name)!r} from {origin[name]}')
            effective[name] = required
            origin[name] = origin.get(name, 'FPGA profile') if origin.get(name) != 'default' else 'FPGA profile'
        if effective['GGML_GEMMINI_ACTIVATION_QUANT'] != 'EXSIA' and effective['GGML_GEMMINI_ENABLE_RMD'] != 'OFF':
            raise BuildConfigurationError(f'{profile_label} non-EXSIA activation requires GGML_GEMMINI_ENABLE_RMD=OFF, matching main')
        if not effective.get('IM2P_SIM_ROOT'):
            raise BuildConfigurationError('FPGA_UART requires IM2P_SIM_ROOT pointing to the selected SCU source')
    option = effective.get('GGML_GEMMINI_OPTION', 'CPU').upper()
    if option not in ('CPU', 'WS') or (option == 'CPU' and backend != 'HARDWARE'):
        raise BuildConfigurationError(f'Illegal Gemmini option/backend combination: {option}+{backend}')
    # Valid option changes reconfigure the existing build tree. CMake replaces
    # backend targets and compile/link commands with the resolved selection.
    if effective.get('CYCLE_DETAIL') == '1' and effective.get('LOG_CYCLE') != '1':
        raise BuildConfigurationError('CYCLE_DETAIL=1 requires LOG_CYCLE=1')
    if effective.get('GGML_GEMMINI_EXSIA_PROFILE_SCOPE', 'OFF') != 'OFF' and effective.get('CYCLE_DETAIL') != '1':
        raise BuildConfigurationError('ExSIA profiling requires CYCLE_DETAIL=1')
    # Alias must not survive as a second contradictory -D after resolution.
    passthrough = [arg for arg in passthrough if not re.match(r'-DIM2P_DIM(?::[^=]+)?=', arg)]
    return effective, origin, passthrough, dry_run


def main():
    if sys.argv[1:2] == ['--validate-hp1-profile']:
        try:
            rmd = normalize('GGML_GEMMINI_ENABLE_RMD', sys.argv[7]) if len(sys.argv) > 7 else 'OFF'
            if rmd not in ('ON', 'OFF'):
                raise BuildConfigurationError('GGML_GEMMINI_ENABLE_RMD must be ON or OFF')
            profile = load_hp1_profile(sys.argv[2], rmd == 'ON')
            requested = tuple(map(int, sys.argv[3:7]))
            actual = (profile['activation_bits'], profile['weight_bits'], profile['dim'], profile['block_size'])
            if requested != actual:
                raise BuildConfigurationError(f'GEMMINI_HP1 CMake/profile mismatch: requested {requested}, profile {actual}')
        except (IndexError, BuildConfigurationError) as error:
            print(f'build configuration error: {error}', file=sys.stderr)
            return 2
        print(profile['profile'])
        return 0
    build_dir, platform, *rest = sys.argv[1:]
    split = rest.index('--')
    defaults = dict(item.split('=', 1) for item in rest[:split])
    defaults.setdefault('GGML_GEMMINI', 'ON')
    try:
        effective, origin, passthrough, dry = resolve(build_dir, platform, defaults, rest[split + 1:], dict(os.environ))
    except ValueError as error:
        print(f'build configuration error: {error}', file=sys.stderr)
        return 2
    summary = {'build_dir': str(Path(build_dir).resolve()), 'precedence': ['command-line', 'environment', 'script-default'],
               'dry_run': dry, 'effective': effective, 'origin': origin,
               'provisioning': 'none; CPU-functional source build' if effective.get('CYCLE_SIM') == '1' else 'matching IM2P_SIM artifacts' if effective.get('GGML_GEMMINI_EXECUTION_BACKEND') == 'IM2P_SIM' else 'none; FPGA_UART uses physical external executor' if effective.get('GGML_GEMMINI_EXECUTION_BACKEND') == 'FPGA_UART' else 'none'}
    print('IM2P_EFFECTIVE_CONFIG=' + json.dumps(summary, sort_keys=True), file=sys.stderr)
    for name, value in effective.items():
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            print(f'{name}_DEFAULT={shlex.quote(value)}')
    cmake_args = passthrough + [f'-D{name}={value}' for name, value in sorted(effective.items()) if configurable(name)]
    print('IM2P_EFFECTIVE_CMAKE_ARGS=(' + ' '.join(map(shlex.quote, cmake_args)) + ')')
    print('IM2P_BUILD_DRY_RUN=' + ('1' if dry else '0'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
