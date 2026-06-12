# Start dashboard + Faith tracking (after enrollment is complete)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not (Test-Path "data\db\face_db.npz")) {
    Write-Host "No enrollment found. Run enroll first:" -ForegroundColor Red
    Write-Host "  python -m src.enroll --name Faith --camera-index 1 --camera-rotate 180 --fresh"
    exit 1
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; .\venv\Scripts\python.exe -m http.server 8765 --directory dashboard"
Start-Sleep -Seconds 1
Start-Process "http://localhost:8765/index.html"

Write-Host "Manual lock: press L on the tracking window. Search starts when Faith leaves frame."
& .\venv\Scripts\python.exe addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Faith
