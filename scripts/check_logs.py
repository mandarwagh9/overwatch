"""Check full Jetson logs + verify tracking mode."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _jetson_common import connect, get_credentials

creds = get_credentials()
print(f"Connecting to {creds['host']}...")
c = connect(creds)


def remote_run(cmd):
    print(f'> {cmd}')
    si, so, se = c.exec_command(cmd, timeout=30)
    o = so.read().decode('utf-8', errors='replace')
    e = se.read().decode('utf-8', errors='replace')
    if o.strip():
        print(o.strip())
    if e.strip():
        print('ERR:', e.strip()[:500])

print('=== FULL LOGS ===')
remote_run('cat /tmp/overwatch.log')

print('\n=== SCIPY CHECK ===')
remote_run('python3 -c "from scipy.optimize import linear_sum_assignment; print(\'scipy OK\')"')

print('\n=== PERCEPTION PIPELINE CHECK ===')
remote_run('python3 -c "import sys; sys.path.insert(0, \'/home/mandar/OVERWATCH/backend\'); from app.core.perception_pipeline import PerceptionPipeline; print(\'Pipeline import OK\')"')

c.close()
