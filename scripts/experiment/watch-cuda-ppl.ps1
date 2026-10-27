# Read-only foreground monitor; Ctrl+C closes this display only.
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$LinuxUser = 'mojoi',
    [string]$LinuxRoot = '/mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini',
    [string]$Config = 'scripts/experiment/cuda-ppl-jobs.rtx3070.json',
    [string]$Output = '/home/mojoi/potal-results/rtx3070-full-24',
    [switch]$Once,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'
$wslArguments = @('-d', $Distro, '-u', $LinuxUser, '--cd', $LinuxRoot, '--exec',
    'python3', 'scripts/experiment/watch-cuda-ppl.py', '--config', $Config, '--output', $Output)
if ($Once) { $wslArguments += '--once' }
if ($Json) { $wslArguments += '--json' }
& wsl.exe @wslArguments
exit $LASTEXITCODE
