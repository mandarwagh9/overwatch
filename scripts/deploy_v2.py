"""Deploy Overwatch v2.0 (multi-agent perception) to Jetson Orin Nano."""
import paramiko, time, os

JETSON_IP = '192.168.1.12'
JETSON_USER = 'mandar'
JETSON_PASS = 'mandar'
REMOTE_BASE = '/home/mandar/overwatch-backend'

# All files to upload (local -> remote)
FILES = {
    # Core perception pipeline (NEW)
    r"backend\app\core\perception_pipeline.py": "app/core/perception_pipeline.py",
    # Modified backend files
    r"backend\app\core\detection_engine.py": "app/core/detection_engine.py",
    r"backend\app\core\tracking_manager.py": "app/core/tracking_manager.py",
    r"backend\app\core\world_model.py": "app/core/world_model.py",
    r"backend\app\core\camera_manager.py": "app/core/camera_manager.py",
    r"backend\app\api\websocket_handler.py": "app/api/websocket_handler.py",
    r"backend\main.py": "main.py",
    r"backend\app\config.py": "app/config.py",
    r"backend\app\__init__.py": "app/__init__.py",
    r"backend\app\core\__init__.py": "app/core/__init__.py",
    r"backend\app\api\__init__.py": "app/api/__init__.py",
    r"backend\requirements.txt": "requirements.txt",
}

LOCAL_ROOT = r"C:\OVERWATCH"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(JETSON_IP, username=JETSON_USER, password=JETSON_PASS, timeout=15)
print(f"✅ Connected to Jetson at {JETSON_IP}")

def run(cmd, t=300):
    print(f'  > {cmd[:140]}')
    si, so, se = c.exec_command(cmd, timeout=t)
    if 'nohup' in cmd and '&' in cmd:
        time.sleep(2)
        return '', '', 0
    o = so.read().decode('utf-8', errors='replace')
    e = se.read().decode('utf-8', errors='replace')
    rc = so.channel.recv_exit_status()
    if o.strip():
        for line in o.strip().split('\n')[:30]:
            print(f'    {line}')
    if e.strip() and rc != 0:
        for line in e.strip().split('\n')[:10]:
            print(f'    WARN: {line}')
    return o, e, rc

# 1. Kill old backend
print("\n🛑 Stopping old backend...")
run("pkill -f 'python3 main.py' 2>/dev/null || true; sleep 2")

# 2. Upload all files
print("\n📤 Uploading files...")
sftp = c.open_sftp()
for local_rel, remote_rel in FILES.items():
    local_path = os.path.join(LOCAL_ROOT, local_rel)
    remote_path = f"{REMOTE_BASE}/{remote_rel}"
    if not os.path.exists(local_path):
        print(f"  ⚠️  SKIP (not found): {local_rel}")
        continue
    sftp.put(local_path, remote_path)
    size = os.path.getsize(local_path)
    print(f"  ✅ {remote_rel} ({size:,} bytes)")
sftp.close()

# 3. Install new dependencies
print("\n📦 Installing new dependencies (scipy, PyJWT)...")
run(f"cd {REMOTE_BASE} && pip3 install scipy PyJWT 2>&1 | tail -5")

# 4. Verify imports
print("\n🔍 Verifying imports...")
run(f"cd {REMOTE_BASE} && python3 -c \"import scipy; import jwt; from app.core.perception_pipeline import PerceptionPipeline; print('All imports OK')\"")

# 5. Check .env
print("\n📋 Current .env:")
run(f"cat {REMOTE_BASE}/.env")

# 6. Start backend
print("\n🚀 Starting backend...")
run(f"cd {REMOTE_BASE} && nohup python3 main.py > /tmp/overwatch.log 2>&1 & echo STARTED")

# 7. Wait for startup
print("\n⏳ Waiting 15s for startup...")
time.sleep(15)

# 8. Check logs
print("\n📜 Startup log:")
run("cat /tmp/overwatch.log")

# 9. Process check
print("\n🔍 Process check:")
run("ps aux | grep 'main.py' | grep -v grep")

# 10. HTTP health check
print("\n🌐 Health check:")
run("curl -sk https://localhost:8000/ 2>&1")

c.close()
print("\n✅ Deploy complete!")
