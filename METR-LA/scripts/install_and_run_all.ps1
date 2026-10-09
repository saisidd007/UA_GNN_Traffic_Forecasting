<#
Install dependencies into `.venv` (if needed) and run the project's run_all wrapper.

Usage (PowerShell, from repo root):
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\scripts\install_and_run_all.ps1

This script will:
 - create a virtualenv at `.venv` if missing
 - activate it for the current session
 - install `-r requirements.txt`
 - if installing `torch` fails, attempt a CPU-only wheel fallback
 - run `.
un_all.ps1`

Note: This script attempts safe fallbacks but cannot guarantee binary compatibility
for `torch` on all Windows setups. If installation fails, follow official PyTorch
instructions at https://pytorch.org/get-started/locally/.
#>

param()

function Run-Command($cmd) {
    Write-Host "`n> $cmd" -ForegroundColor Cyan
    $proc = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile -Command $cmd" -Wait -PassThru -NoNewWindow
    return $proc.ExitCode
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtualenv at .venv..." -ForegroundColor Yellow
    $ec = Run-Command "python -m venv .venv"
    if ($ec -ne 0) { Write-Host "Failed to create venv (exit $ec). Ensure Python is installed and on PATH." -ForegroundColor Red; exit $ec }
}

Write-Host "Activate the venv in this session now:" -ForegroundColor Yellow
Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "After activating, run this script again or run: .\run_all.ps1" -ForegroundColor Yellow

# Attempt to install requirements (this runs in the current shell, not the activated venv)
Write-Host "\nAttempting to install requirements into the venv..." -ForegroundColor Yellow
Write-Host "Note: You should first run: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow

Write-Host "Running: .venv\Scripts\Activate.ps1; pip install -r requirements.txt" -ForegroundColor Cyan
$cmd = ".venv\Scripts\Activate.ps1; pip install -r requirements.txt"
$ec = Run-Command $cmd
if ($ec -eq 0) {
    Write-Host "Requirements installed successfully." -ForegroundColor Green
} else {
    Write-Host "Requirements install failed with exit code $ec." -ForegroundColor Red
    Write-Host "Attempting CPU-only PyTorch fallback..." -ForegroundColor Yellow
    $cmd2 = ".venv\Scripts\Activate.ps1; pip uninstall -y torch torchvision torchaudio; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    $ec2 = Run-Command $cmd2
    if ($ec2 -eq 0) {
        Write-Host "CPU PyTorch wheel installed successfully." -ForegroundColor Green
    } else {
        Write-Host "CPU PyTorch fallback also failed (exit $ec2)." -ForegroundColor Red
        Write-Host "Please follow the official guide: https://pytorch.org/get-started/locally/" -ForegroundColor Red
        exit $ec2
    }
}

Write-Host "\nAll installs complete — running run_all wrapper now." -ForegroundColor Green
$runAll = Join-Path $root 'run_all.ps1'
if (Test-Path $runAll) {
    Write-Host "Invoking $runAll" -ForegroundColor Cyan
    $ec3 = Run-Command ". '$runAll'"
    if ($ec3 -ne 0) { Write-Host "run_all wrapper failed with exit $ec3" -ForegroundColor Red; exit $ec3 }
} else {
    Write-Host 'run_all.ps1 not found. Exiting.' -ForegroundColor Red
    Write-Host $runAll -ForegroundColor Yellow
    exit 1
}
