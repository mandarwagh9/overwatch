"""
Overwatch → Jetson Orin Nano deployment script.
Handles: SSH, file transfer, dependency install, TensorRT export, backend launch.
"""

import paramiko
import os
import sys
import time
import stat

JETSON_HOST = "192.168.1.4"
JETSON_USER = "mandar"
JETSON_PASS = "mandar"
LOCAL_PROJECT = r"C:\OVERWATCH"
REMOTE_DIR = "/home/mandar/OVERWATCH"

def ssh_connect():
    """Create SSH connection to Jetson."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"🔌 Connecting to {JETSON_USER}@{JETSON_HOST}...")
    client.connect(JETSON_HOST, username=JETSON_USER, password=JETSON_PASS, timeout=15)
    print("✅ SSH connected!")
    return client

def run_cmd(client, cmd, print_output=True, timeout=300):
    """Run a command via SSH, return (stdout, stderr, exit_code)."""
    print(f"  ▶ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    # For background commands, don't block waiting for output
    if '&' in cmd and 'nohup' in cmd:
        time.sleep(2)
        return "", "", 0
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    exit_code = stdout.channel.recv_exit_status()
    if print_output and out.strip():
        for line in out.strip().split('\n'):
            print(f"    {line}")
    if print_output and err.strip():
        for line in err.strip().split('\n'):
            print(f"    ⚠ {line}")
    return out, err, exit_code

def upload_project(client):
    """Upload the OVERWATCH project to the Jetson via SFTP."""
    print("\n📦 Uploading project files to Jetson...")
    sftp = client.open_sftp()

    # Directories and files to upload (skip node_modules, __pycache__, .git, etc)
    skip_dirs = {'node_modules', '__pycache__', '.git', '.next', 'dist', 'build', '.venv', 'venv'}
    skip_files = {'.env'}  # We'll create .env separately on the Jetson

    def ensure_remote_dir(remote_path):
        """Recursively create remote directory."""
        dirs_to_create = []
        check = remote_path
        while check and check != '/':
            try:
                sftp.stat(check)
                break
            except FileNotFoundError:
                dirs_to_create.append(check)
                check = os.path.dirname(check)
        for d in reversed(dirs_to_create):
            try:
                sftp.mkdir(d)
            except Exception:
                pass

    uploaded = 0
    # Upload backend and certs
    for subdir in ['backend', 'certs', 'scripts']:
        local_sub = os.path.join(LOCAL_PROJECT, subdir)
        if not os.path.exists(local_sub):
            continue
        for root, dirs, files in os.walk(local_sub):
            # Filter out skip dirs
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            rel = os.path.relpath(root, LOCAL_PROJECT).replace('\\', '/')
            remote_path = f"{REMOTE_DIR}/{rel}"
            ensure_remote_dir(remote_path)

            for f in files:
                if f in skip_files:
                    continue
                local_file = os.path.join(root, f)
                remote_file = f"{remote_path}/{f}"
                try:
                    sftp.put(local_file, remote_file)
                    uploaded += 1
                    if uploaded % 10 == 0:
                        print(f"    📄 {uploaded} files uploaded...")
                except Exception as e:
                    print(f"    ⚠ Failed to upload {rel}/{f}: {e}")

    # Also upload README
    try:
        sftp.put(os.path.join(LOCAL_PROJECT, "README.md"), f"{REMOTE_DIR}/README.md")
        uploaded += 1
    except Exception:
        pass

    sftp.close()
    print(f"  ✅ {uploaded} files uploaded to {REMOTE_DIR}")

def main():
    print("=" * 60)
    print("🚀 OVERWATCH → Jetson Orin Nano Deployment")
    print("=" * 60)

    client = ssh_connect()

    # ── Step 1: System info ──
    print("\n📋 Step 1: Checking Jetson system...")
    run_cmd(client, "hostname; uname -m; cat /etc/nv_tegra_release 2>/dev/null || echo 'N/A'")
    run_cmd(client, "python3 --version")
    
    # Check for CUDA
    out, _, _ = run_cmd(client, "python3 -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')\" 2>&1")
    has_torch_cuda = "CUDA: True" in out
    
    if not has_torch_cuda:
        print("  ⚠ PyTorch with CUDA not detected. Checking if torch is installed at all...")
        run_cmd(client, "pip3 list 2>/dev/null | grep -i torch || echo 'torch not installed'")

    # ── Step 2: Upload project ──
    print("\n📋 Step 2: Uploading project...")
    run_cmd(client, f"mkdir -p {REMOTE_DIR}")
    upload_project(client)

    # ── Step 3: Install dependencies ──
    print("\n📋 Step 3: Installing dependencies...")
    run_cmd(client, f"cd {REMOTE_DIR}/backend && pip3 install -r requirements-jetson.txt --break-system-packages 2>&1 | tail -5", timeout=600)

    # ── Step 4: Check if yolov8n.pt exists, download if not ──
    print("\n📋 Step 4: Ensuring YOLOv8n model exists...")
    out, _, code = run_cmd(client, f"ls -la {REMOTE_DIR}/backend/yolov8n.pt 2>/dev/null")
    if code != 0:
        print("  Downloading yolov8n.pt...")
        run_cmd(client, f"cd {REMOTE_DIR}/backend && python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\" 2>&1 | tail -3", timeout=120)

    # ── Step 5: Export TensorRT engine (if CUDA available) ──
    print("\n📋 Step 5: TensorRT engine export...")
    out, _, code = run_cmd(client, f"ls -la {REMOTE_DIR}/backend/yolov8n.engine 2>/dev/null")
    if code == 0:
        print("  ✅ yolov8n.engine already exists, skipping export")
    elif has_torch_cuda:
        # Install onnx first (required by ultralytics for TRT export pipeline)
        print("  📦 Installing onnx (required for TensorRT export)...")
        run_cmd(client, "pip3 install onnx onnxruntime --break-system-packages 2>&1 | tail -3", timeout=120)
        print("  ⏳ Exporting TensorRT FP16 engine (this takes ~2-5 min on Orin Nano)...")
        run_cmd(client, 
            f"cd {REMOTE_DIR}/backend && python3 -c \""
            f"from ultralytics import YOLO; "
            f"model = YOLO('yolov8n.pt'); "
            f"model.export(format='engine', half=True, imgsz=640, device=0)"
            f"\" 2>&1 | tail -10",
            timeout=600)
        # Verify
        run_cmd(client, f"ls -lh {REMOTE_DIR}/backend/yolov8n.engine 2>/dev/null || echo 'Export may have failed'")
    else:
        print("  ⚠ Skipping TensorRT export — CUDA not available")
        print("  Will use yolov8n.pt with CPU/CUDA fallback")

    # ── Step 6: Create .env ──
    print("\n📋 Step 6: Creating backend .env...")
    # Check if TensorRT engine exists to decide model path
    out, _, code = run_cmd(client, f"test -f {REMOTE_DIR}/backend/yolov8n.engine && echo EXISTS", print_output=False)
    if "EXISTS" in out:
        model_path = "yolov8n.engine"
        half = "true"
    else:
        model_path = "yolov8n.pt"
        half = "false" if not has_torch_cuda else "true"

    env_content = f"""# Overwatch Jetson Orin Nano config
MODEL_PATH={model_path}
DEVICE={"cuda:0" if has_torch_cuda else "auto"}
HALF_PRECISION={half}
DETECTION_CLASSES=[0]
CONFIDENCE_THRESHOLD=0.5
SSL_ENABLED=true
SSL_CERTFILE=certs/cert.pem
SSL_KEYFILE=certs/key.pem
HOST=0.0.0.0
PORT=8000
"""
    run_cmd(client, f"cat > {REMOTE_DIR}/backend/.env << 'ENVEOF'\n{env_content}ENVEOF")
    print(f"  ✅ .env created (model={model_path}, device={'cuda:0' if has_torch_cuda else 'auto'})")

    # ── Step 7: Kill any existing backend, start new one ──
    print("\n📋 Step 7: Starting backend...")
    run_cmd(client, "pkill -f 'python3 main.py' 2>/dev/null; sleep 1", print_output=False)
    
    # Start in background with nohup
    run_cmd(client,
        f"cd {REMOTE_DIR}/backend && "
        f"nohup python3 main.py > /tmp/overwatch.log 2>&1 & "
        f"echo \"PID: $!\"")
    
    # Wait and check logs
    time.sleep(5)
    print("\n📋 Backend startup log:")
    run_cmd(client, "head -30 /tmp/overwatch.log")

    # ── Step 8: Verify it's running ──
    print("\n📋 Step 8: Verifying...")
    out, _, _ = run_cmd(client, "curl -sk https://localhost:8000/ 2>&1 | head -5 || echo 'NOT RESPONDING'")
    
    if "NOT RESPONDING" not in out:
        print("\n" + "=" * 60)
        print("🎉 OVERWATCH backend is LIVE on Jetson Orin Nano!")
        print(f"   Backend URL: https://192.168.1.4:8000")
        print(f"   Mobile cam:  https://192.168.1.4:8000/mobile")
        print(f"   Model: {model_path}")
        print("=" * 60)
    else:
        print("\n⚠ Backend may still be starting up. Check with:")
        print(f"   ssh mandar@192.168.1.4 'tail -50 /tmp/overwatch.log'")

    client.close()
    print("\n🔌 SSH disconnected. Done!")

if __name__ == "__main__":
    main()
