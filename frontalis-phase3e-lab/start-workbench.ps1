param([int]$Port = 8903)
$ErrorActionPreference = 'Stop'
$labRoot = $PSScriptRoot
$labPython = Join-Path (Split-Path $labRoot -Parent) 'frontalis-phase3d-lab\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) { throw '既存のPhase 3-D研究用Pythonが見つかりません。CLAUDE_HANDOFF.mdの依存環境を確認してください。' }
if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) {
    $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location -LiteralPath $labRoot
Write-Host "frontalis research workbench: http://127.0.0.1:$Port"
& $labPython -B -u playground/server.py --port $Port
