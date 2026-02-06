"""Force kill and restart Overwatch on Jetson."""
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
    o = so.read().decode('utf-8', errors='replace')
    e = se.read().decode('utf-8', errors='replace')
    if o.strip():
        print(o.strip())
    if e.strip():
        print('ERR:', e.strip()[:400])

# Force kill ALL python3 main.py
print('\n=== KILLING ALL ===')
run('kill -9 5232 2>/dev/null || true')
run("pkill -9 -f 'python3 main.py' 2>/dev/null ; true")
time.sleep(3)

# Verify port free
run('ss -tlnp | grep 8000 || echo PORT_FREE')

# Restart
print('\n=== RESTARTING ===')
run('cd /home/mandar/OVERWATCH/backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 & echo STARTED')

time.sleep(15)
print('\n=== LOGS ===')
run('cat /tmp/overwatch.log')

print('\n=== PROCESS ===')
run('ps aux | grep main.py | grep -v grep')

print('\n=== HEALTH ===')
run('curl -sk https://localhost:8000/ 2>&1 | head -3')

c.close()
print('\nDone!')
