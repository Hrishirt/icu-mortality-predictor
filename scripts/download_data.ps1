# Download PhysioNet Challenge 2012 training data.
$DataDir = Join-Path $PSScriptRoot "..\data\physionet"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

$BaseUrl = "https://physionet.org/files/challenge-2012/1.0.0"

Write-Host "Downloading Outcomes-a.txt..."
curl -L -o (Join-Path $DataDir "Outcomes-a.txt") "$BaseUrl/Outcomes-a.txt"

Write-Host "Downloading set-a.zip..."
curl -L -o (Join-Path $DataDir "set-a.zip") "$BaseUrl/set-a.zip"

Write-Host "Extracting set-a..."
Expand-Archive -Path (Join-Path $DataDir "set-a.zip") -DestinationPath $DataDir -Force

Write-Host "Done. Data available at $DataDir"
