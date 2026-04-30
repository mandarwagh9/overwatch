"""Shared SSH/SFTP helpers for Overwatch deploy scripts.

Credentials are read from environment:
- JETSON_HOST (default 192.168.1.10)
- JETSON_USER (default mandar)
- JETSON_PASS (no default; falls back to getpass)
- JETSON_KEY  (path to private key; preferred over password)
"""
from __future__ import annotations
import getpass
import os
import sys
from typing import Optional, Tuple

import paramiko


def get_credentials() -> dict:
    host = os.environ.get("JETSON_HOST", "192.168.1.10")
    user = os.environ.get("JETSON_USER", "mandar")
    key_path = os.environ.get("JETSON_KEY")
    password = os.environ.get("JETSON_PASS")

    if not key_path and not password:
        try:
            password = getpass.getpass(f"Password for {user}@{host}: ")
        except (EOFError, KeyboardInterrupt):
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    return {"host": host, "user": user, "password": password, "key_path": key_path}


def connect(creds: Optional[dict] = None) -> paramiko.SSHClient:
    if creds is None:
        creds = get_credentials()
    client = paramiko.SSHClient()
    try:
        client.load_system_host_keys()
    except Exception:
        pass
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    kwargs = {"hostname": creds["host"], "username": creds["user"], "timeout": 10}
    if creds.get("key_path"):
        kwargs["key_filename"] = creds["key_path"]
    elif creds.get("password"):
        kwargs["password"] = creds["password"]

    client.connect(**kwargs)
    return client


def run(client: paramiko.SSHClient, cmd: str, check: bool = True) -> Tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}\nstderr: {err}")
    return rc, out, err
