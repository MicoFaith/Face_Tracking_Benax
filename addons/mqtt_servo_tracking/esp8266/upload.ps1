param(
    [Parameter(Mandatory = $true)]
    [string]$Port,

    [string]$Fqbn = "esp8266:esp8266:nodemcuv2"
)

$ErrorActionPreference = "Stop"
$SketchDir = Join-Path $PSScriptRoot "face_tracker_servo"

if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) {
    throw "arduino-cli was not found in PATH. Install it first: https://arduino.github.io/arduino-cli/latest/installation/"
}

Write-Host "Compiling $SketchDir for $Fqbn"
arduino-cli compile --fqbn $Fqbn $SketchDir

Write-Host "Uploading to $Port"
arduino-cli upload -p $Port --fqbn $Fqbn $SketchDir

Write-Host "Done."

