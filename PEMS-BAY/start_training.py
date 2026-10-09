import subprocess
import time
import os

# Change to project directory
os.chdir(r'C:\Users\rockk\OneDrive\Desktop\traffic-flow-gnn')

# Wait for file handles
time.sleep(2)

# Start training
try:
    p = subprocess.Popen(
        [r'C:\venv_traffic_flow\Scripts\python.exe', 'enhanced_train.py', '--epochs', '50'],
        stdout=open('results/train_50.log', 'w'),
        stderr=subprocess.STDOUT
    )
    print(f'[START] Training restarted (PID: {p.pid})')
    print(f'[LOG] Output: results/train_50.log')
except Exception as e:
    print(f'[ERROR] Failed to start training: {e}')
