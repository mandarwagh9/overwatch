#!/usr/bin/env python3
"""
Quick restart script for Overwatch Jetson backend.
Usage: python restart_jetson.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _jetson_common import connect, get_credentials

import time

REMOTE_DIR = "/home/mandar/overwatch"


def main():
    creds = get_credentials()
    host = creds["host"]
    user = creds["user"]
    print(f"Connecting to {host}...")

    client = connect(creds)

    # Kill existing
    print("Stopping existing backend...")
    client.exec_command("pkill -f 'python3 main.py' 2>/dev/null; sleep 2")

    # Start new
    print("Starting backend...")
    client.exec_command(
        f"cd {REMOTE_DIR}/backend && "
        f"nohup python3 main.py > /tmp/overwatch.log 2>&1 &"
    )

    # Wait and check
    time.sleep(10)

    stdin, stdout, stderr = client.exec_command("tail -20 /tmp/overwatch.log")
    print("\nLog:")
    print(stdout.read().decode())

    stdin, stdout, stderr = client.exec_command("curl -sk https://localhost:8000/ | head -3")
    response = stdout.read().decode()

    if "Overwatch" in response or response:
        print("\nBackend is running!")
    else:
        print(f"\nCheck logs: ssh {user}@{host} 'tail -50 /tmp/overwatch.log'")

    client.close()


if __name__ == "__main__":
    main()
