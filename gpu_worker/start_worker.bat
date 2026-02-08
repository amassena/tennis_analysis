@echo off
cd /d C:\Users\amass\tennis_analysis
set PYTHONPATH=C:\Users\amass\tennis_analysis
C:\Users\amass\tennis_analysis\venv\Scripts\python.exe gpu_worker/worker.py --coordinator http://100.115.41.118:8080
