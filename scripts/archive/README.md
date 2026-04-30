# Archived deploy scripts

These scripts duplicated `deploy_jetson.py` / `restart_jetson.py` with drifted
hosts, paths, and hardcoded credentials. They are kept here for reference and
may be removed in a future cleanup.

- `deploy_v2.py` — pointed at `192.168.1.12` and a wrong local root
- `_restart_now.py` / `force_restart.py` — duplicates of `restart_jetson.py`
- `fix_jetson.py` — one-off fix script

Use `scripts/deploy_jetson.py` and `scripts/restart_jetson.py` instead.
Set `JETSON_HOST`, `JETSON_USER`, `JETSON_PASS` (or `JETSON_KEY`) in env.

These archived files still contain hardcoded credentials in git history. The
user should rotate the Jetson password as a one-time hygiene step.
