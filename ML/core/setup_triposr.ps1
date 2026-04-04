<#
PowerShell setup script to clone TripoSR and install dependencies.

Place this file in `ML/core` and run from that directory.
#>

param()

Set-StrictMode -Version Latest

$ErrorActionPreference = 'Stop'

$repo = 'https://github.com/VAST-AI-Research/TripoSR.git'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$target = Join-Path $root 'triposr'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is required but not found in PATH. Please install git."
    exit 1
}

if (-not (Test-Path $target)) {
    Write-Output "Cloning TripoSR into $target..."
    git clone $repo $target
} else {
    Write-Output "TripoSR already cloned; attempting to pull latest changes..."
    Push-Location $target
    try {
        git pull --rebase
    } catch {
        Write-Warning "git pull failed (possibly unstaged changes). Skipping pull."
    }
    Pop-Location
}

# Patch requirements to prefer pymcubes over various torch-mcubes package names
 $reqFiles = Get-ChildItem -Path $target -Recurse -Include requirements*.txt -ErrorAction SilentlyContinue
if ($reqFiles -ne $null -and $reqFiles.Count -gt 0) {
    foreach ($f in $reqFiles) {
        Write-Output "Patching $($f.FullName)"
        (Get-Content $f.FullName) -replace '(?i)torch[-_]?mcubes|tochmcubes', 'pymcubes' | Set-Content $f.FullName
    }
} else {
    Write-Output "No requirements.txt files found to patch."
}

# Create venv in this ml folder
$venv = Join-Path $root '.venv'
if (-not (Test-Path $venv)) {
    Write-Output "Creating virtual environment at $venv"
    python -m venv $venv
}

# Activate and install requirements
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Python executable not found in venv: $python"
    exit 1
}

& $python -m pip install -U pip

if ($reqFiles -ne $null -and $reqFiles.Count -gt 0) {
    foreach ($f in $reqFiles) {
        Write-Output "Installing requirements from $($f.FullName)"
        & $python -m pip install -r $f.FullName
    }
} else {
    Write-Output "No requirements.txt found in TripoSR; skip installing requirements"
}

# Ensure pymcubes is installed and available
& $python -m pip install -U pymcubes

# Copy shims into the cloned repo so imports resolve
$shimSrc = Join-Path $root 'shims\torchmcubes.py'
$shimDestDir = Join-Path $target 'shims'
if (Test-Path $shimSrc) {
    if (-not (Test-Path $shimDestDir)) { New-Item -ItemType Directory -Path $shimDestDir | Out-Null }
    Copy-Item -Path $shimSrc -Destination (Join-Path $shimDestDir 'torchmcubes.py') -Force
    Write-Output "Copied shim to $shimDestDir"
} else {
    Write-Output "Shim not found at $shimSrc; skipping copy."
}

Write-Output "Setup complete. Inspect $target and set TRIPO_COMMAND to run inference. Example:"
Write-Output "$env:TRIPO_COMMAND = \"$python $target\inference.py --input {input_image} --out {output_dir}\""
