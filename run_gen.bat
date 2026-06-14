@echo off
cd /d "C:\Users\9khal\Documents\RocketSurrogate"
echo =========================================
echo  RocketSurrogate Data Generation
echo  Target: 2000  Workers: 6  Balanced
echo  Started: %date% %time%
echo  Output: outputs\rocket_data_2k.jsonl
echo =========================================
echo.
set PYTHONUNBUFFERED=1
python -u src/rocket_sim/generator.py --count 2000 --method random --workers 6 --oversample 3.0 --output outputs/rocket_data_2k.jsonl --splits-dir outputs/splits_2k --plots-dir outputs/plots_2k
echo.
echo =========================================
echo  Done! Exit code: %ERRORLEVEL%
echo  Finished: %date% %time%
echo =========================================
pause