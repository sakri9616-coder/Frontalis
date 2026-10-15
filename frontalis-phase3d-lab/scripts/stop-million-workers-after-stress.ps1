$taskLab = Split-Path -Parent $PSScriptRoot
$pendingWorkers = @(0, 1)
while ($pendingWorkers.Count -gt 0) {
  foreach ($taskWorker in @($pendingWorkers)) {
    $taskLoss = @('bounded_signal', 'bounded_logit')[$taskWorker]
    $taskReport = Join-Path $taskLab "reports\million-stress\raw8\$taskLoss\nn-h31250\seed4311-p0.json"
    if (-not (Test-Path -LiteralPath $taskReport)) { continue }
    $taskSaved = Get-Content -Raw -LiteralPath $taskReport | ConvertFrom-Json
    if ($taskSaved.status -ne 'complete') { throw "Worker $taskWorker stress failed; preserve process for investigation." }
    foreach ($taskSnapshot in @($taskSaved.snapshotAfter1000, $taskSaved.snapshotAfterContrary96)) {
      foreach ($taskExtension in @('.json', '.npz')) {
        if (-not (Test-Path -LiteralPath (Join-Path $taskLab ($taskSnapshot + $taskExtension)))) { throw 'Committed stress report snapshot missing.' }
      }
    }
    # Only the known independent lab worker child interpreters are eligible.
    # A success report is written after both snapshots, before next diagnostic.
    $taskExpectedChildPid = @(38100, 7824)[$taskWorker]
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $taskExpectedChildPid"
    if ($null -ne $taskProcess) {
      if ($taskProcess.Name -ne 'python.exe' -or $taskProcess.CommandLine -notmatch 'numeric\.py\s+million-worker\s+--worker\s+' + $taskWorker + '(\s|$)' -or $taskProcess.ExecutablePath -ne 'C:\Users\sakur\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe') {
        throw "Worker $taskWorker PID identity changed; do not stop."
      }
      Stop-Process -Id $taskExpectedChildPid
      Write-Output "Worker $taskWorker stopped after complete stress save; PID $taskExpectedChildPid."
    } else { Write-Output "Worker $taskWorker already exited after saved stress." }
    $pendingWorkers = @($pendingWorkers | Where-Object { $_ -ne $taskWorker })
  }
  if ($pendingWorkers.Count -gt 0) { Start-Sleep -Seconds 1 }
}
