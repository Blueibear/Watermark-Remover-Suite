# Codex Autonomous Daemon
# Auto-resumes project.yaml builds when Codex stops or exits unexpectedly.
# Author: James (Nasteeshirts)

$projectFile = "project.yaml"
$logFile = "codex_autorun.log"
$delaySeconds = 10

Write-Host "Starting Codex Daemon for $projectFile..." -ForegroundColor Cyan

while ($true) {
    # Read the last known phase from progress checkpoint if present
    $checkpoint = ""
    if (Test-Path "codex_progress.yaml") {
        try {
            $lines = Get-Content "codex_progress.yaml" | Select-String -Pattern "Phase"
            if ($lines) {
                $checkpoint = ($lines[-1].ToString() -replace ".*Phase[:\s-]*", "").Trim()
                Write-Host "Resuming from $checkpoint"
            }
        } catch {
            Write-Warning "Could not parse checkpoint file."
        }
    }

    $resumeArg = ""
    if ($checkpoint) {
        $resumeArg = "--from `"$checkpoint`""
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "`n[$timestamp] Launching Codex..." -ForegroundColor Green
    Add-Content $logFile "`n[$timestamp] Launching Codex from $checkpoint"

    # Run Codex autonomously, logging stdout and stderr
    try {
        & codex follow $projectFile $resumeArg --no-interactive --auto-continue *>> $logFile
    } catch {
        Write-Warning "Codex crashed or exited unexpectedly."
        Add-Content $logFile "Codex crashed or exited unexpectedly at $timestamp"
    }

    Write-Host "`nCodex stopped - restarting in $delaySeconds seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds $delaySeconds
}
