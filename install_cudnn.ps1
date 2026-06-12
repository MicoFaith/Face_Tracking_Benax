# Helper script to install cuDNN DLLs
# Usage: .\install_cudnn.ps1 "C:\path\to\cudnn\extracted\folder"

param(
    [Parameter(Mandatory=$true)]
    [string]$CudnnPath
)

$cudaBinPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"

Write-Host "Looking for cuDNN DLLs in: $CudnnPath" -ForegroundColor Yellow

# Check if path exists
if (-not (Test-Path $CudnnPath)) {
    Write-Host "Error: Path not found: $CudnnPath" -ForegroundColor Red
    exit 1
}

# Find bin folder
$binPath = Join-Path $CudnnPath "bin"
if (-not (Test-Path $binPath)) {
    # Maybe it's directly in the path
    $binPath = $CudnnPath
}

# Find cuDNN DLLs
$dlls = Get-ChildItem $binPath -Filter "*cudnn*.dll" -ErrorAction SilentlyContinue

if ($dlls.Count -eq 0) {
    Write-Host "Error: No cuDNN DLLs found in $binPath" -ForegroundColor Red
    Write-Host "Please check the path and ensure cuDNN DLLs are present." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found $($dlls.Count) cuDNN DLL(s):" -ForegroundColor Green
foreach ($dll in $dlls) {
    Write-Host "  - $($dll.Name)" -ForegroundColor Cyan
}

# Copy DLLs
Write-Host "`nCopying DLLs to: $cudaBinPath" -ForegroundColor Yellow

if (-not (Test-Path $cudaBinPath)) {
    Write-Host "Error: CUDA bin path not found: $cudaBinPath" -ForegroundColor Red
    Write-Host "Please ensure CUDA Toolkit 13.1 is installed." -ForegroundColor Yellow
    exit 1
}

try {
    Copy-Item $dlls.FullName -Destination $cudaBinPath -Force
    Write-Host "Successfully copied DLLs!" -ForegroundColor Green
    Write-Host "`nPlease restart your terminal/Python process for changes to take effect." -ForegroundColor Yellow
} catch {
    Write-Host "Error copying DLLs: $_" -ForegroundColor Red
    Write-Host "You may need to run this script as Administrator." -ForegroundColor Yellow
    exit 1
}
