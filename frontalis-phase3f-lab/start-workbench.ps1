$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot '..\frontalis-phase3d-lab\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Phase 3-D Python runtime is missing. See CLAUDE_HANDOFF.md.' }
if (-not $env:ANTHROPIC_API_KEY) { $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User') }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $runtime -B (Join-Path $PSScriptRoot 'playground\server.py') --port 8905
