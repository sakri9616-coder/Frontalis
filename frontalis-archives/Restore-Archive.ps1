param([Parameter(Mandatory=$true)][string]$Name)
$ErrorActionPreference='Stop'
$workspacePath=Split-Path -Parent $PSScriptRoot
$researchPython=Join-Path $workspacePath 'frontalis-phase3d-lab\.venv\Scripts\python.exe'
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& $researchPython -B (Join-Path $PSScriptRoot 'archive_manager.py') restore $Name
if ($LASTEXITCODE -ne 0) { throw 'Archive restoration failed; existing research was not overwritten.' }
