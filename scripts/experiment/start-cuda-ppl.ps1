# Keep a foreground WSL client alive while the independent user service runs.
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$LinuxUser = 'mojoi',
    [string]$LinuxRoot = '/mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini',
    [string]$Config = 'scripts/experiment/cuda-ppl-jobs.rtx3070.json',
    [string]$Output = '/home/mojoi/potal-results/rtx3070-full-24'
)
$ErrorActionPreference = 'Stop'
$env:POTAL_QUEUE_ROOT = $LinuxRoot
$env:POTAL_QUEUE_CONFIG = $Config
$env:POTAL_QUEUE_OUTPUT = $Output
$env:WSLENV = (($env:WSLENV -split ':' | Where-Object { $_ -and $_ -notmatch '^POTAL_QUEUE_(ROOT|CONFIG|OUTPUT)' }) + @('POTAL_QUEUE_ROOT','POTAL_QUEUE_CONFIG','POTAL_QUEUE_OUTPUT')) -join ':'
$supervisor = Join-Path $PSScriptRoot 'wsl-cuda-ppl-supervisor.sh'
$linuxSupervisor = (& wsl.exe -d $Distro -u $LinuxUser --exec wslpath -a $supervisor).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL supervisor path' }
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $Distro, '-u', $LinuxUser, '--exec', 'bash', $linuxSupervisor) -WindowStyle Hidden -PassThru
Write-Output "WSL anchor PID: $($process.Id); queue output: $Output"
Write-Output 'Inspect queue.log/status.json or run watch-cuda-ppl.py; closing PowerShell does not stop the hidden anchor.'
