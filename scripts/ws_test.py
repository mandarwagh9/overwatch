import sys
import time
import subprocess

url = sys.argv[1] if len(sys.argv) > 1 else 'ws://192.168.1.8:8000/ws'
max_msgs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Ensure websocket-client
try:
    import websocket
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'websocket-client'])
    import websocket

# Ensure msgpack
try:
    import msgpack
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'msgpack'])
    import msgpack

print(f"Connecting to {url} (will read {max_msgs} messages)...")

try:
    ws = websocket.create_connection(url, timeout=10)
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)

count = 0
start = time.time()
while count < max_msgs:
    try:
        data = ws.recv()
    except Exception as e:
        print(f"Recv error: {e}")
        break

    if data is None:
        print("No data (None)")
        break

    if isinstance(data, bytes):
        print(f"[{count}] Binary message — {len(data)} bytes")
        # Try msgpack decode
        try:
            obj = msgpack.unpackb(data, raw=False)
            print(f"    Decoded msgpack type: {type(obj).__name__}")
            # Print small summary
            if isinstance(obj, dict):
                keys = list(obj.keys())[:10]
                print(f"    Keys: {keys}")
            else:
                print(f"    Value (repr): {repr(obj)[:200]}")
        except Exception as e:
            print(f"    msgpack decode failed: {e}")
            print(f"    hex preview: {data[:32].hex()}...")
    else:
        print(f"[{count}] Text message — {len(data)} chars")
        print(f"    {data[:400]}")

    count += 1

print(f"Read {count} messages in {time.time()-start:.2f}s")
ws.close()
