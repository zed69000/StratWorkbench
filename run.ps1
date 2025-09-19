# Launcher (auto-open, avec auto-unblock)
$ErrorActionPreference = "Stop"

# Débloque automatiquement tous les fichiers du dossier courant
try {
    Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File | Unblock-File -ErrorAction SilentlyContinue
} catch {}

# Règle d’exécution (valide uniquement pour ce process)
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force } catch {}

# Déplacement dans le dossier du script
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Création venv si absent
if (!(Test-Path ".venv")) { python -m venv .venv }

$PIP = ".\.venv\Scripts\pip.exe"
$STREAMLIT = ".\.venv\Scripts\streamlit.exe"

# MAJ pip + install requirements
& $PIP install --upgrade pip
& $PIP install -r requirements.txt

# Désactive la télémétrie streamlit
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Lancement de Streamlit sur port choisi
$port = 8501
Start-Process -FilePath $STREAMLIT -ArgumentList @("run","app.py","--server.port","$port","--server.headless","true")

# Attente que le serveur soit prêt
for ($i=0; $i -lt 40; $i++) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$port" -TimeoutSec 2
    if ($r.StatusCode -ge 200) { break }
  } catch {}
  Start-Sleep -Seconds 1
}

# Ouvre le navigateur
Start-Process "http://localhost:$port"
