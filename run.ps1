# Launcher final (auto-detect projectDir, no debug pauses)
$ErrorActionPreference = "Stop"

# Detect project directory automatically (where this script is located)
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

# Unblock files
try {
    Get-ChildItem -LiteralPath $projectDir -Recurse -File | Unblock-File -ErrorAction SilentlyContinue
} catch {}

# Execution policy for this process only
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force } catch {}

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Install it and try again." -ForegroundColor Red
    exit 1
}

# Create venv if missing
if (!(Test-Path "$projectDir\.venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv "$projectDir\.venv"
}

$PIP = "$projectDir\.venv\Scripts\pip.exe"
$STREAMLIT = "$projectDir\.venv\Scripts\streamlit.exe"

# Verify pip
if (!(Test-Path $PIP)) {
    Write-Host "pip not found in venv." -ForegroundColor Red
    exit 1
}

# Upgrade pip
& $PIP install --upgrade pip

# Install requirements
if (Test-Path "$projectDir\requirements.txt") {
    & $PIP install -r "$projectDir\requirements.txt"
}

# Ensure Streamlit is installed
if (!(Test-Path $STREAMLIT)) {
    & $PIP install streamlit
}

# Disable Streamlit telemetry
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Choose port
$port = 8503

# Launch Streamlit in a new PowerShell window (keeps logs open)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& `"$STREAMLIT`" run `"$projectDir\app.py`" --server.port $port --server.headless true"

# Open browser automatically
Start-Sleep -Seconds 3
Start-Process "http://localhost:$port"
