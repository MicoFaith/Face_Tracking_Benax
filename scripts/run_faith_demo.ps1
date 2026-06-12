# FaithLock full workflow: clear -> manual enroll -> track
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_faith_demo.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

Write-Host "=== Step 1: Clear all enrollment data ===" -ForegroundColor Cyan
& .\venv\Scripts\python.exe scripts\reset_speaker_db.py

Write-Host "`n=== Step 2: Enroll Faith (manual — save when ready) ===" -ForegroundColor Cyan
Write-Host "SPACE = save photo | S = finish (8+ photos) | only Faith in frame"
& .\venv\Scripts\python.exe -m src.enroll --name Faith --camera-index 1 --camera-rotate 180 --fresh

if (-not (Test-Path "data\db\face_db.npz")) {
    Write-Host "Enrollment not saved. Run enroll again and press S when done." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 3: Dashboard ===" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; .\venv\Scripts\python.exe -m http.server 8765 --directory dashboard"
Start-Sleep -Seconds 1
Start-Process "http://localhost:8765/index.html"

Write-Host "`n=== Step 4: Lock + follow Faith ===" -ForegroundColor Cyan
Write-Host "Re-flash ESP8266 if servo does not move (faithlock MQTT topic)."
& .\venv\Scripts\python.exe addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Faith
