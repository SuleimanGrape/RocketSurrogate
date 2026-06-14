Write-Host "=================================="
Write-Host "RocketSurrogate Data Generation"
Write-Host "Target: 2000, Workers: 6"
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=================================="
Write-Host ""

$LogFile = "outputs\generation_2000_log.txt"
"=== Generation started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $LogFile -Encoding utf8

Set-Location "C:\Users\9khal\Documents\RocketSurrogate"

python src\rocket_sim\generator.py `
    --count 2000 `
    --method random `
    --balanced `
    --workers 6 `
    --oversample 3.0 `
    --output outputs\rocket_data_2k.jsonl `
    --splits-dir outputs\splits_2k `
    --plots-dir outputs\plots_2k 2>&1 | Tee-Object -FilePath $LogFile -Append

$ExitCode = $LASTEXITCODE
"=== Generation finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') with exit code $ExitCode ===" | Out-File -FilePath $LogFile -Encoding utf8 -Append

Write-Host ""
Write-Host "=================================="
Write-Host "Done! Exit code: $ExitCode"
Write-Host "Log saved to: $LogFile"
Read-Host "Press Enter to close"