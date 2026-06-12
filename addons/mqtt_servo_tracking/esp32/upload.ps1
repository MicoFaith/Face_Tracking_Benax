param(
    [Parameter(Mandatory = $true)]
    [string]$Port,

    [string]$Fqbn = "esp32:esp32:esp32"
)

$ErrorActionPreference = "Stop"
$SketchDir = Join-Path $PSScriptRoot "face_tracker_servo"

if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) {
    throw "arduino-cli was not found in PATH. Install it first: https://arduino.github.io/arduino-cli/latest/installation/"
}

Write-Host "Installing ESP32 core and libraries (if needed)..."
arduino-cli core update-index | Out-Null
arduino-cli core install esp32:esp32 | Out-Null
arduino-cli lib install "PubSubClient" "ESP32Servo" | Out-Null

Write-Host "Compiling $SketchDir for $Fqbn"
arduino-cli compile --fqbn $Fqbn $SketchDir

Write-Host "Uploading to $Port"
arduino-cli upload -p $Port --fqbn $Fqbn $SketchDir

Write-Host "Done."
