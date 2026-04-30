"""Kill old backend and restart on Jetson."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.12', username='mandar', password='mandar', timeout=15)
print('Connected')

def run(cmd):
    si, so, se = c.exec_command(cmd, timeout=60)
    if 'nohup' in cmd and '&' in cmd:
        time.sleep(2)
        return
    o = so.read().decode()
    e = se.read().decode()
    if o.strip():
        print(o.strip())
    if e.strip():
        print('ERR:', e.strip()[:300])

# Kill everything on port 8000
print('\n=== KILLING OLD BACKEND ===')
run('fuser -k 8000/tcp 2>/dev/null || true')
run("pkill -9 -f 'python3 main.py' 2>/dev/null || true")
time.sleep(3)
run('ss -tlnp | grep 8000 || echo PORT_FREE')

# Start fresh
print('\n=== STARTING BACKEND ===')
run('cd /home/mandar/overwatch-backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 & echo STARTED')
time.sleep(15)

print('\n=== LOGS ===')
run('tail -25 /tmp/overwatch.log')

print('\n=== PROCESS ===')
run("ps aux | grep 'main.py' | grep -v grep")

print('\n=== HEALTH CHECK ===')
run('curl -sk https://localhost:8000/ 2>&1 || curl -s http://localhost:8000/ 2>&1')

c.close()
print('\nDone!')
