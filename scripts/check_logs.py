"""Check full Jetson logs + verify tracking mode."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.12', username='mandar', password='mandar', timeout=15)

def run(cmd):
    print(f'> {cmd}')
    si, so, se = c.exec_command(cmd, timeout=30)
    o = so.read().decode('utf-8', errors='replace')
    e = se.read().decode('utf-8', errors='replace')
    if o.strip():
        print(o.strip())
    if e.strip():
        print('ERR:', e.strip()[:500])

print('=== FULL LOGS ===')
run('cat /tmp/overwatch.log')

print('\n=== SCIPY CHECK ===')
run('python3 -c "from scipy.optimize import linear_sum_assignment; print(\'scipy OK\')"')

print('\n=== PERCEPTION PIPELINE CHECK ===')
run('python3 -c "import sys; sys.path.insert(0, \'/home/mandar/OVERWATCH/backend\'); from app.core.perception_pipeline import PerceptionPipeline; print(\'Pipeline import OK\')"')

c.close()
