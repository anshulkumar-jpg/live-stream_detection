<#
  One-command pipeline: checks prerequisites, starts all services, waits until healthy,
  prints status + URLs, then streams the logs of every service.

  .\run.ps1              start everything (GPU worker) and follow logs
  .\run.ps1 -Dev         CPU-only mode (mock models, worker-dev-0 instead of gpu-worker-0)
  .\run.ps1 -Build       rebuild images first
  .\run.ps1 -NoLogs      start, show status, then exit (no log following)
  .\run.ps1 -Down        stop everything
#>
param(
  [switch]$Dev,
  [switch]$Build,
  [switch]$NoLogs,
  [switch]$Down,
  [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

$compose = @("compose", "--env-file", ".env")
if ($Dev) { $compose += @("--profile", "dev") }

if ($Down) {
  Step "Stopping all services"
  docker @compose --profile dev down
  exit $LASTEXITCODE
}

# 1. prerequisites
Step "Checking prerequisites"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "docker not found in PATH" }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker daemon is not running - start Docker Desktop" }
if (-not (Test-Path .env)) {
  if (Test-Path .env.example) { Copy-Item .env.example .env; Fail ".env created from .env.example - fill in VMS_FERNET_KEY and the change-me-* values, then re-run" }
  Fail ".env is missing"
}
if ((Get-Content .env -Raw) -match "change-me") { Write-Host "WARNING: .env still contains change-me-* placeholders" -ForegroundColor Yellow }
New-Item -ItemType Directory -Force models | Out-Null
if (-not $Dev -and -not (Test-Path "models\best.pt")) { Write-Host "WARNING: models\best.pt not found - the GPU worker will exit" -ForegroundColor Yellow }
docker @compose config *> $null
if ($LASTEXITCODE -ne 0) { Fail "docker compose config is invalid (run: docker compose --env-file .env config)" }
Write-Host "ok"

# 2. start
$services = if ($Dev) { @("api", "worker-dev-0", "kafka", "postgres", "mediamtx") } else { @() }
$up = @("up", "-d")
if ($Build) { $up += "--build" }
Step ("Starting services" + $(if ($Dev) { " (dev / CPU mode)" } else { "" }))
docker @compose @up @services
if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed" }

# 3. wait for healthy
Step "Waiting for services to become healthy (timeout ${TimeoutSec}s)"
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$ready = $false
while ((Get-Date) -lt $deadline) {
  try {
    if ((Invoke-WebRequest -Uri http://localhost:8000/ready -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $ready = $true; break }
  } catch { }
  Write-Host "." -NoNewline
  Start-Sleep -Seconds 3
}
Write-Host ""
if ($ready) { Write-Host "API is ready" -ForegroundColor Green }
else { Write-Host "API not ready after ${TimeoutSec}s - check logs below" -ForegroundColor Yellow }

# 4. status
Step "Service status"
docker @compose ps

Step "URLs"
Write-Host "  API docs   http://localhost:8000/docs"
Write-Host "  Viewer     http://localhost:8000/viewer/<camera_id>"
Write-Host "  WebRTC     http://localhost:8889/<camera_id>/"
Write-Host "  HLS        http://localhost:8888/<camera_id>/index.m3u8   (the bare :8888 root is 404 by design)"

# 5. logs
if ($NoLogs) { exit 0 }
Step "Following logs of all services (Ctrl+C stops the log view; services keep running; .\run.ps1 -Down stops them)"
docker @compose logs -f --tail 20
