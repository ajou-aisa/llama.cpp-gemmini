#!/usr/bin/env python3
"""Resolve build-script options before provisioning. No build or device access."""
import json
import os
from pathlib import Path
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from typing import Final
from evaluation_build_options import validate_evaluation_options

class BuildConfigurationError(ValueError):
    pass

SHELL_CONTROLS: Final = frozenset((
    'BUILD_DIR', 'BUILD_JOBS', 'IM2P_ARTIFACT_SET', 'IM2P_CACHE_JOBS',
    'APPLE_SILICON_ARCH', 'LIBOMP_PREFIX',
))

def configurable(name: str) -> bool:
    return name not in SHELL_CONTROLS and (
        name.startswith(('GGML_', 'IM2P_', 'LOG_', 'CYCLE_', 'CMAKE_', 'LLAMA_'))
        or name in ('BUILD_SHARED_LIBS', 'OpenMP_ROOT', 'GEMMINI_SW_PATH'))


def normalize(name: str, value: str) -> str:
    if name == 'GGML_CPU_CYCLE_LOG':
        if value.upper() in ('OFF', 'NO', 'FALSE', 'N', '0', '', 'IGNORE', 'NOTFOUND') or value.upper().endswith('-NOTFOUND'):
            return '0'
        return '1'
    if name in ('GGML_GEMMINI', 'GGML_GEMMINI_ENABLE_RMD', 'GGML_GEMMINI_DEQUANT_FP_TEST'):
        if value.upper() in ('ON', 'YES', 'TRUE', '1'):
            return 'ON'
        if value.upper() in ('OFF', 'NO', 'FALSE', '0'):
            return 'OFF'
    return value


def resolve(build_dir: str, platform: str, defaults: Mapping[str, str],
            args: Sequence[str], environment: Mapping[str, str]
            ) -> tuple[dict[str, str], dict[str, str], dict[str, str], list[str], bool]:
    cli, cli_types, passthrough, dry_run = {}, {}, [], False
    for arg in args:
        if arg == '--dry-run':
            dry_run = True
        elif arg.startswith('-D'):
            match = re.fullmatch(r'-D([A-Za-z_][A-Za-z0-9_]*)(?::(BOOL|FILEPATH|PATH|STRING|INTERNAL))?=(.*)', arg, re.S)
            if not match:
                raise BuildConfigurationError(f'Unsupported -D syntax: {arg}; use -DNAME[:TYPE]=value')
            cli[match[1]] = match[3]
            cli_types[match[1]] = match[2] or ''
        elif arg in ('-Wdev', '-Wno-dev', '-Werror=dev', '-Wno-error=dev', '--warn-uninitialized', '--debug-find'):
            passthrough.append(arg)
        else:
            raise BuildConfigurationError(f'Unsupported script configure argument: {arg}; use direct CMake for presets, toolchains, -C/-U/-S/-B/-G')
    env = {k: v for k, v in environment.items() if configurable(k) and not k.endswith('_DEFAULT')}
    env_defaults = {k[:-8]: v for k, v in environment.items()
                    if k.endswith('_DEFAULT') and configurable(k[:-8])}
    script_defaults = {k: v for k, v in defaults.items()
                       if configurable(k) and k not in env_defaults}
    # The alias is resolved at its own precedence level; same-level disagreement is an error.
    for level in (script_defaults, env_defaults, env, cli):
        if 'IM2P_DIM' in level:
            if 'GGML_GEMMINI_DIM' in level and level['IM2P_DIM'] != level['GGML_GEMMINI_DIM']:
                raise BuildConfigurationError('IM2P_DIM and GGML_GEMMINI_DIM must match when set at the same precedence level')
            level['GGML_GEMMINI_DIM'] = level.pop('IM2P_DIM')
    if 'IM2P_DIM' in cli_types:
        cli_types.setdefault('GGML_GEMMINI_DIM', cli_types.pop('IM2P_DIM'))
    effective = dict(script_defaults)
    origin = {key: 'script-default' for key in script_defaults}
    for label, level in (('default-environment', env_defaults),
                         ('environment', env), ('command-line', cli)):
        effective.update(level)
        origin.update({key: label for key in level})
    if origin.get('GGML_CPU_CYCLE_LOG', 'script-default') == 'script-default':
        effective['GGML_CPU_CYCLE_LOG'] = effective.get('LOG_CYCLE', '0')
        origin['GGML_CPU_CYCLE_LOG'] = 'derived-default:LOG_CYCLE'
    effective = {key: normalize(key, value) for key, value in effective.items()}
    validate_evaluation_options(effective)
    if effective.get('GGML_GEMMINI_ACT_METRICS') == '1':
        effective['GGML_GEMMINI_ACT_QUANT_METRICS'] = '1'
        origin['GGML_GEMMINI_ACT_QUANT_METRICS'] = 'enabled:GGML_GEMMINI_ACT_METRICS'
    if 'GGML_GEMMINI' not in effective:
        raise BuildConfigurationError('The build script must declare GGML_GEMMINI explicitly')
    if platform != 'build-riscv.sh' and effective.get('CMAKE_TOOLCHAIN_FILE'):
        raise BuildConfigurationError('Native build scripts do not provision cross toolchains; use direct CMake with explicit target artifacts')
    if effective.get('CYCLE_DETAIL') == '1' and effective.get('LOG_CYCLE') != '1':
        raise BuildConfigurationError('CYCLE_DETAIL=1 requires LOG_CYCLE=1')
    profile = effective.get('GGML_GEMMINI_EXSIA_PROFILE_SCOPE', 'OFF').upper()
    if profile not in ('OFF', 'TIMELINE', 'STAGE'):
        raise BuildConfigurationError('GGML_GEMMINI_EXSIA_PROFILE_SCOPE must be OFF, TIMELINE, or STAGE')
    if profile != 'OFF' and effective.get('CYCLE_DETAIL') != '1':
        raise BuildConfigurationError('ExSIA profiling requires CYCLE_DETAIL=1')
    if 'GGML_GEMMINI_EXSIA_PROFILE_SCOPE' in effective:
        effective['GGML_GEMMINI_EXSIA_PROFILE_SCOPE'] = profile
    backend = effective.get('GGML_GEMMINI_EXECUTION_BACKEND', 'HARDWARE')
    if backend == 'FPGA_UART':
        raise BuildConfigurationError('FPGA_UART is deprecated and unsupported; use HARDWARE or IM2P_SIM')
    if effective['GGML_GEMMINI'] == 'OFF':
        effective.update(GGML_GEMMINI_EXECUTION_BACKEND='HARDWARE', IM2P_SIM_IMPLEMENTATION='')
        origin.update({name: 'disabled:GGML_GEMMINI' for name in
                       ('GGML_GEMMINI_EXECUTION_BACKEND', 'IM2P_SIM_IMPLEMENTATION')})
        return effective, origin, cli_types, passthrough, dry_run
    cycle_sim = effective.get('CYCLE_SIM', '0')
    if backend not in ('HARDWARE', 'IM2P_SIM'):
        raise BuildConfigurationError(f'Unknown GGML_GEMMINI_EXECUTION_BACKEND={backend}')
    if platform == 'build-riscv.sh' and backend != 'HARDWARE' and cycle_sim != '1':
        raise BuildConfigurationError('build-riscv.sh is the HARDWARE lane; IM2P_SIM uses the native x86/ARM64 script')
    implementation = effective.get('IM2P_SIM_IMPLEMENTATION', 'LEGACY_BSV')
    if implementation not in ('GEMMINI_HP1', 'LEGACY_BSV'):
        raise BuildConfigurationError(f'Unknown IM2P_SIM_IMPLEMENTATION={implementation}')
    if backend != 'IM2P_SIM':
        if origin.get('IM2P_SIM_IMPLEMENTATION') in ('default-environment', 'environment', 'command-line'):
            raise BuildConfigurationError(
                'IM2P_SIM_IMPLEMENTATION is valid only with '
                'GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM')
        effective['IM2P_SIM_IMPLEMENTATION'] = ''
        origin['IM2P_SIM_IMPLEMENTATION'] = 'cleared:non-IM2P_SIM'
    else:
        effective['IM2P_SIM_IMPLEMENTATION'] = implementation
        if 'IM2P_SIM_IMPLEMENTATION' not in origin:
            origin['IM2P_SIM_IMPLEMENTATION'] = 'IM2P_SIM default'
        cache_dir = effective.get('IM2P_SIM_BUILD_DIR') or str(Path(effective.get('IM2P_SIM_ROOT', '')) / 'build')
        effective['IM2P_SIM_BUILD_DIR'] = str(Path(cache_dir).resolve())
    option = effective.get('GGML_GEMMINI_OPTION', 'CPU').upper()
    if option not in ('CPU', 'WS') or (option == 'CPU' and backend != 'HARDWARE'):
        raise BuildConfigurationError(f'Illegal Gemmini option/backend combination: {option}+{backend}')
    return effective, origin, cli_types, passthrough, dry_run


def main() -> int:
    build_dir, platform, *rest = sys.argv[1:]
    split = rest.index('--')
    defaults = dict(item.split('=', 1) for item in rest[:split])
    try:
        effective, origin, cli_types, passthrough, dry = resolve(build_dir, platform, defaults, rest[split + 1:], dict(os.environ))
        controls = {name: os.environ.get(name) or os.environ.get(name + '_DEFAULT') or value
                    for name, value in defaults.items() if name in SHELL_CONTROLS}
        controls['BUILD_DIR'] = build_dir
        artifact_set = controls.get('IM2P_ARTIFACT_SET', 'SELECTED')
        if artifact_set not in ('SELECTED', 'ALL_MATCHED'):
            raise BuildConfigurationError('IM2P_ARTIFACT_SET must be SELECTED or ALL_MATCHED')
        if 'IM2P_CACHE_JOBS' in controls and not controls['IM2P_CACHE_JOBS']:
            controls['IM2P_CACHE_JOBS'] = '1' if artifact_set == 'ALL_MATCHED' else controls['BUILD_JOBS']
        for name in ('BUILD_JOBS', 'IM2P_CACHE_JOBS'):
            if name in controls and not re.fullmatch(r'[1-9][0-9]*', controls[name]):
                raise BuildConfigurationError(f'{name} must be a positive integer')
    except ValueError as error:
        print(f'build configuration error: {error}', file=sys.stderr)
        return 2
    cmake_args = passthrough + [f'-D{name}{":" + cli_types[name] if cli_types.get(name) else ""}={value}'
                                for name, value in sorted(effective.items())]
    summary = {'build_dir': str(Path(build_dir).resolve()), 'platform': platform,
               'precedence': ['command-line', 'environment', 'default-environment', 'script-default'],
               'dry_run': dry, 'effective': effective, 'origin': origin,
               'controls': controls, 'cmake_args': cmake_args,
               'provisioning': 'none; CPU-functional source build' if effective.get('CYCLE_SIM') == '1' else 'matching IM2P_SIM artifacts' if effective.get('GGML_GEMMINI_EXECUTION_BACKEND') == 'IM2P_SIM' else 'none'}
    print('IM2P_EFFECTIVE_CONFIG=' + json.dumps(summary, sort_keys=True), file=sys.stderr)
    for name, value in effective.items():
        if configurable(name) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            print(f'{name}_DEFAULT={shlex.quote(value)}')
    for name, value in controls.items():
        print(f'{name}_DEFAULT={shlex.quote(value)}')
    print('IM2P_EFFECTIVE_CMAKE_ARGS=(' + ' '.join(map(shlex.quote, cmake_args)) + ')')
    print('IM2P_BUILD_DRY_RUN=' + ('1' if dry else '0'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
