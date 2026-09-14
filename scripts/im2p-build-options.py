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
IDENTITY = tuple(FPGA) + ('GGML_GEMMINI_EXECUTION_BACKEND',)


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
                raise ValueError(f'Unsupported -D syntax: {arg}; use -DNAME[:TYPE]=value')
            cli[match[1]] = match[3]
            passthrough.append(arg)
        elif arg in ('-Wdev', '-Wno-dev', '-Werror=dev', '-Wno-error=dev', '--warn-uninitialized', '--debug-find'):
            passthrough.append(arg)
        else:
            raise ValueError(f'Unsupported script configure argument: {arg}; use direct CMake for presets, toolchains, -C/-U/-S/-B/-G')
    if platform != 'build-riscv.sh' and ('CMAKE_TOOLCHAIN_FILE' in cli or environment.get('CMAKE_TOOLCHAIN_FILE')):
        raise ValueError('Native build scripts do not provision cross toolchains; use direct CMake with explicit target artifacts')

    cache = {}
    cache_path = Path(build_dir) / 'CMakeCache.txt'
    if cache_path.is_file():
        for line in cache_path.read_text().splitlines():
            match = re.fullmatch(r'([^#/:][^:]*):(BOOL|FILEPATH|PATH|STRING)=(.*)', line)
            if match and configurable(match[1]):
                cache[match[1]] = match[3]
    if platform != 'build-riscv.sh' and cache.get('CMAKE_TOOLCHAIN_FILE'):
        raise ValueError('Native build script found a cross-toolchain cache; use a new BUILD_DIR and direct CMake')
    env = {k: v for k, v in environment.items() if configurable(k) and not k.endswith('_DEFAULT')}
    for key, value in environment.items():
        if key.endswith('_DEFAULT') and configurable(key[:-8]):
            env.setdefault(key[:-8], value)
    # The alias is resolved at its own precedence level; same-level disagreement is an error.
    for level in (defaults, cache, env, cli):
        if 'IM2P_DIM' in level:
            if 'GGML_GEMMINI_DIM' in level and level['IM2P_DIM'] != level['GGML_GEMMINI_DIM']:
                raise ValueError('IM2P_DIM and GGML_GEMMINI_DIM must match when set at the same precedence level')
            level['GGML_GEMMINI_DIM'] = level.pop('IM2P_DIM')
    effective = dict(defaults)
    origin = {key: 'default' for key in defaults}
    for label, level in (('cache', cache), ('environment', env), ('command-line', cli)):
        effective.update(level)
        origin.update({key: label for key in level})
    # Backend selected by the script is lane intent, not a reusable cache
    # default. An explicit environment/CLI selection may override it; a stale
    # cache may not silently switch lanes and trigger provisioning.
    if ('GGML_GEMMINI_EXECUTION_BACKEND' not in env and
            'GGML_GEMMINI_EXECUTION_BACKEND' not in cli and
            'GGML_GEMMINI_EXECUTION_BACKEND' in defaults):
        effective['GGML_GEMMINI_EXECUTION_BACKEND'] = defaults['GGML_GEMMINI_EXECUTION_BACKEND']
        origin['GGML_GEMMINI_EXECUTION_BACKEND'] = 'default'
    if origin.get('GGML_CPU_CYCLE_LOG', 'default') == 'default':
        effective['GGML_CPU_CYCLE_LOG'] = effective.get('LOG_CYCLE', '0')
        origin['GGML_CPU_CYCLE_LOG'] = 'derived-default:LOG_CYCLE'
    effective = {key: normalize(key, value) for key, value in effective.items()}
    backend = effective.get('GGML_GEMMINI_EXECUTION_BACKEND', 'HARDWARE')
    if backend not in ('HARDWARE', 'IM2P_SIM', 'FPGA_UART'):
        raise ValueError(f'Unknown GGML_GEMMINI_EXECUTION_BACKEND={backend}')
    if platform == 'build-riscv.sh' and backend != 'HARDWARE':
        raise ValueError('build-riscv.sh is the HARDWARE lane; FPGA_UART uses the native x86/ARM64 script')
    if backend == 'FPGA_UART':
        if effective.get('GGML_GEMMINI_FPGA_SIM_MANIFEST'):
            raise ValueError('GGML_GEMMINI_FPGA_SIM_MANIFEST is invalid for FPGA_UART; physical external executor uses no simulator archive')
        for name, required in FPGA.items():
            if name == 'GGML_GEMMINI_ACTIVATION_QUANT' and origin.get(name, 'default') != 'default':
                effective[name] = effective[name].upper()
                if effective[name] not in ('EXSIA', 'TENSOR', 'TOKEN', 'BLOCK', 'STRIPE'):
                    raise ValueError('FPGA_UART activation must be EXSIA, TENSOR, TOKEN, BLOCK, or STRIPE')
                continue
            if name == 'GGML_GEMMINI_ENABLE_RMD' and origin.get(name, 'default') != 'default':
                if effective.get(name) not in ('ON', 'OFF'):
                    raise ValueError('GGML_GEMMINI_ENABLE_RMD must be ON or OFF')
                continue
            if origin.get(name, 'default') != 'default' and effective.get(name) != required:
                raise ValueError(f'FPGA_UART requires {name}={required}; got {effective.get(name)!r} from {origin[name]}')
            effective[name] = required
            origin[name] = origin.get(name, 'FPGA profile') if origin.get(name) != 'default' else 'FPGA profile'
        if effective['GGML_GEMMINI_ACTIVATION_QUANT'] != 'EXSIA' and effective['GGML_GEMMINI_ENABLE_RMD'] != 'OFF':
            raise ValueError('FPGA_UART non-EXSIA activation requires GGML_GEMMINI_ENABLE_RMD=OFF, matching main')
        if not effective.get('IM2P_SIM_ROOT'):
            raise ValueError('FPGA_UART requires IM2P_SIM_ROOT pointing to the selected SCU source')
    option = effective.get('GGML_GEMMINI_OPTION', 'CPU').upper()
    if option not in ('CPU', 'WS') or (option == 'CPU' and backend != 'HARDWARE'):
        raise ValueError(f'Illegal Gemmini option/backend combination: {option}+{backend}')
    for name in IDENTITY:
        if name in cache and name in effective and normalize(name, cache[name]) != effective[name]:
            raise ValueError(f'Stale build cache: {name} changed from {cache[name]} to {effective[name]}; use a new BUILD_DIR')
    if effective.get('CYCLE_DETAIL') == '1' and effective.get('LOG_CYCLE') != '1':
        raise ValueError('CYCLE_DETAIL=1 requires LOG_CYCLE=1')
    if effective.get('GGML_GEMMINI_EXSIA_PROFILE_SCOPE', 'OFF') != 'OFF' and effective.get('CYCLE_DETAIL') != '1':
        raise ValueError('ExSIA profiling requires CYCLE_DETAIL=1')
    # Alias must not survive as a second contradictory -D after resolution.
    passthrough = [arg for arg in passthrough if not re.match(r'-DIM2P_DIM(?::[^=]+)?=', arg)]
    return effective, origin, passthrough, dry_run


def main():
    build_dir, platform, *rest = sys.argv[1:]
    split = rest.index('--')
    defaults = dict(item.split('=', 1) for item in rest[:split])
    defaults.setdefault('GGML_GEMMINI', 'ON')
    try:
        effective, origin, passthrough, dry = resolve(build_dir, platform, defaults, rest[split + 1:], dict(os.environ))
    except ValueError as error:
        print(f'build configuration error: {error}', file=sys.stderr)
        return 2
    summary = {'build_dir': str(Path(build_dir).resolve()), 'precedence': ['command-line', 'environment', 'cache', 'default'],
               'dry_run': dry, 'effective': effective, 'origin': origin,
               'provisioning': 'matching IM2P_SIM cache' if effective.get('GGML_GEMMINI_EXECUTION_BACKEND') == 'IM2P_SIM' else 'none; FPGA_UART uses physical external executor' if effective.get('GGML_GEMMINI_EXECUTION_BACKEND') == 'FPGA_UART' else 'none'}
    print('IM2P_EFFECTIVE_CONFIG=' + json.dumps(summary, sort_keys=True), file=sys.stderr)
    for name, value in effective.items():
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            print(f'{name}_DEFAULT={shlex.quote(value)}')
    # Emit every resolved value once, after legacy script defaults. Typed user -D is
    # kept first so CMake retains its declared cache type; the effective value wins.
    cmake_args = passthrough + [f'-D{name}={value}' for name, value in sorted(effective.items()) if configurable(name)]
    print('IM2P_EFFECTIVE_CMAKE_ARGS=(' + ' '.join(map(shlex.quote, cmake_args)) + ')')
    print('IM2P_BUILD_DRY_RUN=' + ('1' if dry else '0'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
