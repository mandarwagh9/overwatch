#!/usr/bin/env python3
"""
Overwatch → Jetson Deployment Script
=====================================
Deploys the current Overwatch backend to Jetson Orin Nano.
Supports full deployment, incremental updates, and TensorRT optimization.

Usage:
    python deploy_jetson.py              # Full deployment
    python deploy_jetson.py --update     # Incremental file sync only
    python deploy_jetson.py --tensorrt   # Export TensorRT engine only
"""

import paramiko
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

# Configuration
JETSON_HOST = "192.168.1.10"
JETSON_USER = "mandar"
JETSON_PASS = "mandar"
REMOTE_DIR = "/home/mandar/overwatch"

# Local project root
LOCAL_PROJECT = Path(__file__).parent.parent.resolve()

# Directories/files to upload (relative to LOCAL_PROJECT)
UPLOAD_PATTERNS = [
    "backend/",
    "scripts/",
    "frontend/build/",
    "certs/",
    "README.md",
]

# Include model files
if (LOCAL_PROJECT / "backend" / "yolov8n.pt").exists():
    UPLOAD_PATTERNS.append("backend/yolov8n.pt")
if (LOCAL_PROJECT / "backend" / "yolov8n.engine").exists():
    UPLOAD_PATTERNS.append("backend/yolov8n.engine")

# Files to skip during upload
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".next", "dist", "build", ".pytest_cache"
}
SKIP_FILES = {".env", "*.pyc", "*.pyo"}


class JetsonDeployer:
    def __init__(self, host: str, user: str, password: str):
        self.host = host
        self.user = user
        self.password = password
        self.client: Optional[paramiko.SSHClient] = None
        self.sftp = None
        
    def connect(self) -> None:
        """Establish SSH connection to Jetson."""
        print(f"\n{'='*60}")
        print(f"🔌 Connecting to {self.user}@{self.host}...")
        print(f"{'='*60}")
        
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            self.host, 
            username=self.user, 
            password=self.password, 
            timeout=30,
            banner_timeout=30
        )
        print("✅ SSH connected!")
        
    def disconnect(self) -> None:
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        print("\n🔌 SSH disconnected.")
    
    def run(self, cmd: str, timeout: int = 300, quiet: bool = False) -> Tuple[str, str, int]:
        """Run command via SSH."""
        if not quiet:
            print(f"  ▶ {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
        
        stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        
        # Handle background commands
        if '&' in cmd and 'nohup' in cmd:
            time.sleep(1)
            return "", "", 0
            
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        rc = stdout.channel.recv_exit_status()
        
        if not quiet:
            for line in out.strip().split('\n')[:20]:
                if line.strip():
                    print(f"    {line}")
            if err.strip() and rc != 0:
                for line in err.strip().split('\n')[:10]:
                    if line.strip():
                        print(f"    ⚠ {line}")
        
        return out, err, rc
    
    def check_system(self) -> dict:
        """Check Jetson system info."""
        print("\n📋 Checking Jetson system...")
        
        info = {}
        
        # Hostname
        out, _, _ = self.run("hostname", quiet=True)
        info['hostname'] = out.strip()
        
        # JetPack version
        out, _, _ = self.run("cat /etc/nv_tegra_release 2>/dev/null | head -1", quiet=True)
        info['jetpack'] = out.strip() if out.strip() else "Unknown"
        
        # Python version
        out, _, _ = self.run("python3 --version", quiet=True)
        info['python'] = out.strip()
        
        # CUDA available
        out, _, rc = self.run("python3 -c 'import torch; print(torch.cuda.is_available())'", quiet=True)
        info['cuda'] = rc == 0 and "True" in out
        
        # TensorRT available
        out, _, rc = self.run("python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null", quiet=True)
        info['tensorrt'] = rc == 0
        
        print(f"    Hostname: {info.get('hostname', 'N/A')}")
        print(f"    JetPack: {info.get('jetpack', 'N/A')}")
        print(f"    Python: {info.get('python', 'N/A')}")
        print(f"    CUDA: {'✅' if info.get('cuda') else '❌'}")
        print(f"    TensorRT: {'✅' if info.get('tensorrt') else '❌'}")
        
        return info
    
    def upload_files(self) -> int:
        """Upload project files to Jetson via SFTP."""
        print(f"\n📦 Uploading project files to {REMOTE_DIR}...")
        
        self.sftp = self.client.open_sftp()
        
        # Ensure remote directory exists
        self.run(f"mkdir -p {REMOTE_DIR}", quiet=True)
        
        uploaded = 0
        
        for pattern in UPLOAD_PATTERNS:
            local_path = LOCAL_PROJECT / pattern
            
            if not local_path.exists():
                print(f"  ⚠️  Skipping (not found): {pattern}")
                continue
            
            if local_path.is_file():
                # Get relative path from LOCAL_PROJECT
                rel = os.path.relpath(local_path, LOCAL_PROJECT).replace('\\', '/')
                self._upload_file(local_path, rel)
                uploaded += 1
            else:
                # Directory
                uploaded += self._upload_directory(local_path, pattern)
        
        self.sftp.close()
        print(f"  ✅ {uploaded} files uploaded")
        
        return uploaded
    
    def _ensure_remote_dir(self, remote_path: str) -> None:
        """Ensure remote directory exists."""
        parts = remote_path.split('/')
        for i in range(1, len(parts) + 1):
            path = '/'.join(parts[:i])
            try:
                self.sftp.stat(path)
            except FileNotFoundError:
                try:
                    self.sftp.mkdir(path)
                except:
                    pass
    
    def _upload_file(self, local_path: Path, remote_rel: str) -> None:
        """Upload a single file."""
        remote_path = f"{REMOTE_DIR}/{remote_rel}"
        try:
            self._ensure_remote_dir(os.path.dirname(remote_path))
            self.sftp.put(str(local_path), remote_path)
            print(f"    ✅ {remote_rel}")
        except Exception as e:
            print(f"    ❌ {remote_rel}: {e}")
    
    def _upload_directory(self, local_dir: Path, pattern: str) -> int:
        """Upload a directory recursively."""
        uploaded = 0
        
        for root, dirs, files in os.walk(local_dir):
            # Filter skip dirs
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            
            rel = os.path.relpath(root, LOCAL_PROJECT).replace('\\', '/')
            remote_base = f"{REMOTE_DIR}/{rel}"
            
            # Ensure remote dir
            try:
                self._ensure_remote_dir(remote_base)
            except:
                pass
            
            for f in files:
                # Skip skip files
                if any(f.endswith(ext.replace('*', '')) for ext in SKIP_FILES if '*' in ext):
                    continue
                if f in SKIP_FILES:
                    continue
                    
                local_file = os.path.join(root, f)
                remote_file = f"{remote_base}/{f}"
                
                try:
                    self.sftp.put(local_file, remote_file)
                    uploaded += 1
                    if uploaded % 20 == 0:
                        print(f"    📄 {uploaded} files uploaded...")
                except Exception as e:
                    print(f"    ⚠ {rel}/{f}: {e}")
        
        return uploaded
    
    def install_dependencies(self) -> None:
        """Install Python dependencies on Jetson."""
        print("\n📦 Installing dependencies...")
        
        req_file = f"{REMOTE_DIR}/backend/requirements-jetson.txt"
        
        out, err, rc = self.run(
            f"cd {REMOTE_DIR}/backend && "
            f"pip3 install -r requirements-jetson.txt --break-system-packages 2>&1 | tail -20",
            timeout=600
        )
        
        if rc != 0:
            print("  ⚠️  Some dependencies may have failed. Check output above.")
        else:
            print("  ✅ Dependencies installed")
    
    def export_tensorrt(self, local_model_exists: bool = False) -> str:
        """Export YOLOv8n to TensorRT engine."""
        print("\n🎯 Checking TensorRT engine...")
        
        # Check if engine already exists on Jetson (was uploaded)
        out, _, rc = self.run(f"ls -la {REMOTE_DIR}/backend/yolov8n.engine 2>/dev/null", quiet=True)
        if rc == 0:
            print("  ✅ yolov8n.engine already exists (uploaded)")
            return "yolov8n.engine"
        
        # Check for yolov8n.pt on Jetson
        out, _, rc = self.run(f"ls -la {REMOTE_DIR}/backend/yolov8n.pt 2>/dev/null", quiet=True)
        
        if rc != 0:
            # Model not on Jetson - download it (this is slow)
            print("  📥 Downloading yolov8n.pt (this is slow)...")
            self.run(
                f"cd {REMOTE_DIR}/backend && "
                f"python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"",
                timeout=300
            )
        
        # Check CUDA
        has_cuda = self._check_cuda()
        
        if not has_cuda:
            print("  ⚠️  No CUDA, using CPU PyTorch model")
            return "yolov8n.pt"
        
        # Export to TensorRT (only if CUDA available)
        print("  ⏳ Exporting TensorRT (2-5 min on Orin)...")
        self.run(
            f"cd {REMOTE_DIR}/backend && "
            f"python3 -c \""
            f"from ultralytics import YOLO; "
            f"model = YOLO('yolov8n.pt'); "
            f"model.export(format='engine', half=True, imgsz=640, device=0)"
            f"\" 2>&1 | tail -10",
            timeout=600
        )
        
        # Verify
        out, _, rc = self.run(f"ls -lh {REMOTE_DIR}/backend/yolov8n.engine 2>/dev/null", quiet=True)
        if rc == 0:
            print(f"  ✅ TensorRT ready: {out.strip()}")
            return "yolov8n.engine"
        
        print("  ⚠️  Export failed, using PyTorch model")
        return "yolov8n.pt"
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        out, _, rc = self.run("python3 -c 'import torch; print(torch.cuda.is_available())'", quiet=True)
        return rc == 0 and "True" in out
    
    def create_env(self, model_path: str, has_cuda: bool) -> None:
        """Create/update .env configuration file."""
        print("\n📝 Creating backend configuration...")
        
        half = "true" if has_cuda and "engine" in model_path else "false"
        
        env_content = f"""# Overwatch Jetson Configuration
# Generated by deploy_jetson.py

# Model
MODEL_PATH={model_path}
DEVICE={"cuda:0" if has_cuda else "cpu"}
HALF_PRECISION={half}

# Detection
DETECTION_CLASSES=[0]  # Person only
CONFIDENCE_THRESHOLD=0.4
IOU_THRESHOLD=0.45

# Tracking
TRACKING_MAX_AGE=30
TRACKING_MIN_HITS=3
TRACKING_IOU_THRESHOLD=0.25

# Camera
TARGET_FPS=24
MOBILE_CAMERA_FPS=15
MOBILE_CAMERA_MAX_WIDTH=640

# Network
HOST=0.0.0.0
PORT=8000
SSL_ENABLED=true
SSL_CERTFILE=certs/cert.pem
SSL_KEYFILE=certs/key.pem
"""
        
        self.run(f"cat > {REMOTE_DIR}/backend/.env << 'ENVEOF'\n{env_content}ENVEOF", quiet=True)
        print("  ✅ Configuration created")
    
    def start_backend(self) -> bool:
        """Start the Overwatch backend."""
        print("\n🚀 Starting Overwatch backend...")
        
        # Kill existing process
        self.run("pkill -f 'python3 main.py' 2>/dev/null; sleep 2", quiet=True)
        
        # Start in background
        self.run(
            f"cd {REMOTE_DIR}/backend && "
            f"nohup python3 main.py > /tmp/overwatch.log 2>&1 & "
            f"echo 'PID: $!'"
        )
        
        # Wait for startup
        print("  ⏳ Waiting for startup...")
        time.sleep(8)
        
        # Check logs
        print("\n📜 Backend log:")
        self.run("tail -30 /tmp/overwatch.log")
        
        # Health check
        out, _, rc = self.run(
            "curl -sk https://localhost:8000/ 2>&1 | head -5",
            quiet=True
        )
        
        if rc == 0 and ("Overwatch" in out or "overwatch" in out.lower()):
            print(f"\n{'='*60}")
            print("🎉 OVERWATCH BACKEND IS LIVE!")
            print(f"{'='*60}")
            print(f"   🌐 URL: https://{self.host}:8000")
            print(f"   📱 Mobile: https://{self.host}:8000/mobile")
            return True
        else:
            print("\n⚠️  Backend may still be starting. Check logs with:")
            print(f"   ssh {self.user}@{self.host} 'tail -50 /tmp/overwatch.log'")
            return False
    
    def deploy(self, update_only: bool = False, tensorrt_only: bool = False) -> None:
        """Run full deployment."""
        try:
            self.connect()
            
            # System check
            sys_info = self.check_system()
            
            # Check local model files
            local_model = (LOCAL_PROJECT / "backend" / "yolov8n.pt").exists()
            local_engine = (LOCAL_PROJECT / "backend" / "yolov8n.engine").exists()
            
            if local_engine:
                print("\n📦 Local TensorRT engine found - will upload")
            elif local_model:
                print("\n📦 Local YOLO model found - will upload")
            
            # Upload files
            if not tensorrt_only:
                self.upload_files()
                
                # Install dependencies
                if not update_only:
                    self.install_dependencies()
            
            # TensorRT export
            model_path = self.export_tensorrt(local_model or local_engine)
            
            # Create config
            self.create_env(model_path, sys_info.get('cuda', False))
            
            # Start backend (unless only doing tensorrt export)
            if not tensorrt_only and not update_only:
                self.start_backend()
                
        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Deploy Overwatch to Jetson")
    parser.add_argument("--update", action="store_true", 
                        help="Incremental update (skip dependency install)")
    parser.add_argument("--tensorrt", action="store_true",
                        help="Export TensorRT engine only")
    parser.add_argument("--host", default=JETSON_HOST,
                        help=f"Jetson IP address (default: {JETSON_HOST})")
    args = parser.parse_args()
    
    # Set UTF-8 encoding for Windows compatibility
    import sys
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    print(f"\n{'#'*60}")
    print(f"# OVERWATCH -> JETSON DEPLOYMENT")
    print(f"# Target: {args.host}")
    print(f"{'#'*60}")
    
    deployer = JetsonDeployer(args.host, JETSON_USER, JETSON_PASS)
    deployer.deploy(update_only=args.update, tensorrt_only=args.tensorrt)


if __name__ == "__main__":
    main()
