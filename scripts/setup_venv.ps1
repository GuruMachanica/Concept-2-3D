# PowerShell script to create a single .venv at the repo root and install requirements
# Run this from the repo root. This script is defensive and will not fail if the
# repo-root .venv does not yet exist.

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path -Path (Join-Path $scriptRoot "..")
$venvPath = Join-Path $repoRoot ".venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating venv at: $venvPath"
    python -m venv "$venvPath"
} else {
    Write-Host "Using existing venv at: $venvPath"
}

# Activate and upgrade packaging tools, then install requirements
$activate = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activate) {
    & "$activate"
} else {
    Write-Host "Activation script not found at $activate — continuing without activation"
}

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r (Join-Path $repoRoot "requirements.txt")

Write-Host ".venv created/updated at: $venvPath"
Write-Host "To activate, run the activation script below:"
Write-Host (Join-Path $venvPath "Scripts\Activate.ps1")