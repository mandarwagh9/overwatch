"""Fix and restart Overwatch on Jetson."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.4', username='mandar', password='mandar', timeout=15)
print("Connected!")

def run(cmd, t=120):
    print(f'> {cmd[:120]}')
    si, so, se = c.exec_command(cmd, timeout=t)
    if 'nohup' in cmd and '&' in cmd:
        time.sleep(2)
        return '', '', 0
    o = so.read().decode('utf-8', errors='replace')
    e = se.read().decode('utf-8', errors='replace')
    rc = so.channel.recv_exit_status()
    if o.strip():
        print(o.strip())
    if e.strip():
        print('WARN:', e.strip()[:500])
    return o, e, rc

# 1. Check last logs
print("\n=== LAST LOGS ===")
run("tail -50 /tmp/overwatch.log")

# 2. Upload fixed files
print("\n=== UPLOADING FIXED FILES ===")
sftp = c.open_sftp()
sftp.put(r"C:\OVERWATCH\backend\app\config.py", "/home/mandar/OVERWATCH/backend/app/config.py")
sftp.put(r"C:\OVERWATCH\backend\app\core\detection_engine.py", "/home/mandar/OVERWATCH/backend/app/core/detection_engine.py")
sftp.close()
print("Uploaded config.py + detection_engine.py")

# 3. Check .env
print("\n=== .env CONTENTS ===")
run("cat /home/mandar/OVERWATCH/backend/.env")

# 4. Kill old backend
print("\n=== KILLING OLD BACKEND ===")
run("pkill -f 'python3 main.py' 2>/dev/null || true; sleep 2")

# 5. Restart
print("\n=== STARTING BACKEND ===")
run("cd /home/mandar/OVERWATCH/backend && nohup python3 main.py > /tmp/overwatch.log 2>&1 & echo STARTED_PID=$!")

# 6. Wait and check
time.sleep(10)
print("\n=== STARTUP LOG ===")
run("cat /tmp/overwatch.log")

# 7. Verify process
print("\n=== PROCESS CHECK ===")
run("ps aux | grep 'main.py' | grep -v grep")

# 8. Verify HTTP
print("\n=== HTTP CHECK ===")
run("curl -sk https://localhost:8000/ 2>&1 | head -5")

c.close()
print("\nDone!")
