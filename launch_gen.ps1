try {
    $p = Start-Process -FilePath python -ArgumentList 'src\rocket_sim\generator.py --count 2000 --method random --balanced --workers 6 --oversample 3.0 --output outputs\rocket_data_2k.jsonl --splits-dir outputs\splits_2k --plots-dir outputs\plots_2k' -WorkingDirectory 'C:\Users\9khal\Documents\RocketSurrogate' -WindowStyle Hidden -PassThru -ErrorAction Stop
    Write-Host "Started PID: $($p.Id)"
    $p.Id | Out-File 'C:\Users\9khal\Documents\RocketSurrogate\outputs\gen_pid.txt'
} catch {
    Write-Host "Error: $($Error[0])"
    $Error[0].ToString() | Out-File 'C:\Users\9khal\Documents\RocketSurrogate\outputs\gen_error.txt'
}