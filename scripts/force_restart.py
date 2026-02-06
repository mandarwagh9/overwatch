"""Force kill port 8000, restart backend on Jetson."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.4', username='mandar', password='mandar', timeout=15)
print('Connected')

def run(cmd, t=60):
    print(f'> {cmd}')
    si, so, se = c.exec_command(cmd, timeout=t)
    if 'nohup' in cmd and '&' in cmd:
        time.sleep(2)
        return
    o = so.read().decode()
    e = se.read().decode()
    if o.strip():
        print(o.strip()[:500])
    if e.strip():
        print('ERR:', e.strip()[:300])

# Kill everything on port 8000
run('fuser -k 8000/tcp 2>/dev/null || true')
time.sleep(3)
run("pkill -9 -f 'python3 main.py' 2>/dev/null || true")
time.sleep(2)
run('ss -tlnp | grep 8000 || echo PORT_FREE')

# Start fresh
print('\nStarting backend...')
run('cd /home/mandar/OVERWATCH/backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 & echo STARTED')
time.sleep(15)
run('tail -20 /tmp/overwatch.log')
run('curl -sk https://localhost:8000/ 2>&1 | head -2')
c.close()
print('Done')
