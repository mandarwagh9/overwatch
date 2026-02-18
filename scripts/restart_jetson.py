#!/usr/bin/env python3
"""
Quick restart script for Overwatch Jetson backend.
Usage: python restart_jetson.py
"""

import paramiko
import time

JETSON_HOST = "192.168.1.12"
JETSON_USER = "mandar"
JETSON_PASS = "mandar"
REMOTE_DIR = "/home/mandar/overwatch"

def main():
    print(f"🔌 Connecting to {JETSON_HOST}...")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(JETSON_HOST, username=JETSON_USER, password=JETSON_PASS, timeout=15)
    
    # Kill existing
    print("🛑 Stopping existing backend...")
    client.exec_command("pkill -f 'python3 main.py' 2>/dev/null; sleep 2")
    
    # Start new
    print("🚀 Starting backend...")
    client.exec_command(
        f"cd {REMOTE_DIR}/backend && "
        f"nohup python3 main.py > /tmp/overwatch.log 2>&1 &"
    )
    
    # Wait and check
    time.sleep(10)
    
    stdin, stdout, stderr = client.exec_command("tail -20 /tmp/overwatch.log")
    print("\n📜 Log:")
    print(stdout.read().decode())
    
    stdin, stdout, stderr = client.exec_command("curl -sk https://localhost:8000/ | head -3")
    response = stdout.read().decode()
    
    if "Overwatch" in response or response:
        print("\n✅ Backend is running!")
    else:
        print("\n⚠️ Check logs: ssh mandar@192.168.1.12 'tail -50 /tmp/overwatch.log'")
    
    client.close()

if __name__ == "__main__":
    main()
