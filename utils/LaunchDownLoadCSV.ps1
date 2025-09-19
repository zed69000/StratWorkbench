# run_download.ps1
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (!(Test-Path ".venv")) { python -m venv .venv }

$PIP = ".\.venv\Scripts\pip.exe"
$STREAMLIT = ".\.venv\Scripts\streamlit.exe"

& $PIP install --upgrade pip
& $PIP install streamlit python-binance pandas

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

Start-Process -FilePath $STREAMLIT -ArgumentList @("run","download_binance_app.py","--server.port","8502","--server.headless","true")
for ($i=0; $i -lt 40; $i++) {
  try { $r = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8502" -TimeoutSec 2; if ($r.StatusCode -ge 200) { break } } catch {}
  Start-Sleep -Seconds 1
}
Start-Process "http://localhost:8502"
