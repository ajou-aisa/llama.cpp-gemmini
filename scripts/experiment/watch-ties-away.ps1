param([switch]$Once)
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'watch-full-q4.py'
$linuxScript = (& wsl.exe -d Ubuntu-24.04 -u mojoi --exec wslpath -a $script).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve ties-away monitor path' }
$mode = if ($Once) { '--status' } else { '--watch' }
& wsl.exe -d Ubuntu-24.04 -u mojoi --exec /home/mojoi/.venvs/potal-sensitivity/bin/python $linuxScript $mode
