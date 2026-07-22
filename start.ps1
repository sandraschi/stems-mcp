param([switch]$BackendOnly, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $PSCommandPath
$BackendPort = 11126
$FrontendPort = 11127

Write-Host "=== stems-mcp ===" -ForegroundColor Cyan

# Clear port zombies
Get-NetTCPConnection -LocalPort $BackendPort -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Get-NetTCPConnection -LocalPort $FrontendPort -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

# Start backend
$env:MCP_PORT = "$BackendPort"
$BackendJob = Start-Job -Name "stems-backend" -ScriptBlock {
    param($Root, $Port)
    Set-Location $Root
    uv run python -m stems_mcp --mode http --host 127.0.0.1 --port $Port
} -ArgumentList $ScriptRoot, $BackendPort

# Readiness poll
Write-Host "Waiting for backend on port $BackendPort ..." -ForegroundColor Yellow
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) { Write-Host "Backend ready" -ForegroundColor Green; break }
    } catch {}
    Start-Sleep 1
}

if (-not $BackendOnly) {
    $WebRoot = Join-Path $ScriptRoot "webapp"
    if (Test-Path "$WebRoot\package.json") {
        Start-Process -NoNewWindow -FilePath "npx" -ArgumentList "vite --port $FrontendPort --host" -WorkingDirectory $WebRoot
        Start-Sleep 2
        if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$FrontendPort" }
    }
}

# Keep-alive
try {
    while ($true) {
        if ($BackendJob.State -eq "Completed" -or $BackendJob.State -eq "Failed") {
            Receive-Job $BackendJob
            break
        }
        Start-Sleep 2
    }
} finally {
    Stop-Job $BackendJob -ErrorAction SilentlyContinue
    Remove-Job $BackendJob -ErrorAction SilentlyContinue
}
