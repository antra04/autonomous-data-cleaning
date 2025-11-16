# ==========================================================
# FIX_VENV.PS1 — Clean, rebuild, and reinstall your venv safely
# Author: Antra (Autonomous Data Cleaning Project)
# ==========================================================

# --- CONFIG ---
$venvPath = "venv"
$pythonVersion = "python"   # Change to python3 or python312 if needed
$requirements = @(
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "fpdf2",
    "lightgbm",
    "xgboost"
)

Write-Host "🧹 Cleaning up existing virtual environment..." -ForegroundColor Cyan

# --- STOP ALL PYTHON PROCESSES TO UNLOCK FILES ---
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# --- REMOVE OLD VENV ---
if (Test-Path $venvPath) {
    Write-Host "Deleting old venv folder..."
    try {
        Remove-Item -Recurse -Force $venvPath
        Write-Host "✅ Old venv removed successfully."
    } catch {
        Write-Host "⚠️ Could not delete some files. Try closing VS Code or rebooting, then re-run this script."
    }
} else {
    Write-Host "No existing venv found — continuing..."
}

# --- CREATE NEW VENV ---
Write-Host "🚀 Creating new virtual environment..." -ForegroundColor Cyan
& $pythonVersion -m venv $venvPath
if (!(Test-Path "$venvPath\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment creation failed." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Virtual environment created successfully!" -ForegroundColor Green

# --- ACTIVATE VENV ---
Write-Host "Activating virtual environment..."
& "$venvPath\Scripts\Activate.ps1"

# --- UPGRADE CORE TOOLS ---
Write-Host "⬆️ Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

# --- INSTALL DEPENDENCIES ---
Write-Host "📦 Installing required dependencies..." -ForegroundColor Cyan
foreach ($pkg in $requirements) {
    Write-Host "Installing $pkg..." -ForegroundColor Yellow
    & "$venvPath\Scripts\python.exe" -m pip install --no-cache-dir $pkg
}

Write-Host ""
Write-Host "✅ All dependencies installed successfully!" -ForegroundColor Green
Write-Host ""

# --- CREATE REQUIREMENTS.TXT ---
Write-Host "🧾 Writing requirements.txt..." -ForegroundColor Cyan
$requirements -join "`n" | Out-File -Encoding utf8 "requirements.txt"
Write-Host "✅ requirements.txt created." -ForegroundColor Green

# --- FINAL INSTRUCTIONS ---
Write-Host ""
Write-Host "To start your app, run these:" -ForegroundColor Cyan
Write-Host "--------------------------------------------------"
Write-Host ('& "{0}\Scripts\Activate.ps1"' -f $venvPath) -ForegroundColor Yellow
Write-Host "streamlit run app/streamlit_app.py --server.port 8501" -ForegroundColor Yellow
Write-Host "--------------------------------------------------"
Write-Host ""
Write-Host "🎉 All done! If PowerShell feels slow, reboot once for faster performance." -ForegroundColor Green
Write-Host "==========================================================="
