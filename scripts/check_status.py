"""Quick WebSocket test against Jetson backend."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _jetson_common import connect, get_credentials, run

import json

creds = get_credentials()
print(f"Connecting to {creds['host']}...")
c = connect(creds)


def remote_run(cmd):
    si, so, se = c.exec_command(cmd, timeout=30)
    o = so.read().decode('utf-8', errors='replace')
    e = se.read().decode('utf-8', errors='replace')
    return o.strip(), e.strip()

# Check latest logs
print('=== LATEST LOGS (last 30 lines) ===')
o, e = remote_run('tail -30 /tmp/overwatch.log')
print(o)
if e:
    print('ERR:', e[:300])

# Check API endpoints
print('\n=== API ROOT ===')
o, e = remote_run('curl -sk https://localhost:8000/')
try:
    data = json.loads(o)
    print(json.dumps(data, indent=2))
except:
    print(o)

# Check active connections (rough proxy)
print('\n=== ACTIVE CONNECTIONS ===')
o, e = remote_run('ss -tnp | grep 8000 | head -10')
print(o or 'No connections yet')

# Memory & GPU
print('\n=== GPU STATUS ===')
o, e = remote_run('tegrastats --interval 1000 --count 1 2>/dev/null || echo "tegrastats not available"')
print(o[:300] if o else 'N/A')

c.close()
print('\nDone!')
