@echo off
REM GPU Worker loop - restarts after each job to pick up code changes

cd /d C:\Users\amass\tennis_analysis
set PYTHONPATH=C:\Users\amass\tennis_analysis

:loop
echo [%date% %time%] Starting worker...
C:\Users\amass\tennis_analysis\venv\Scripts\python.exe gpu_worker/worker.py --coordinator http://100.115.41.118:8080 --once

REM Wait 5 seconds before restarting (prevents tight loop if no jobs)
timeout /t 5 /nobreak >nul
goto loop
