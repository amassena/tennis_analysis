@echo off
cd /d C:\Users\Andrew\tennis_analysis
set PYTHONPATH=C:\Users\Andrew\tennis_analysis
C:\Users\Andrew\tennis_analysis\venv\Scripts\python.exe gpu_worker/worker.py --coordinator http://100.115.41.118:8080
