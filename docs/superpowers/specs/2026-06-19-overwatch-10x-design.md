# OVERWATCH 10× — Design & Roadmap

**Date:** 2026-06-19
**Status:** Active (Phase A in progress)
**Author:** brainstormed with Claude Code

## Goal

Close the gap between what OVERWATCH *claims* and what it *does*, on a base that is
provably tested. Today the README and `ARCHITECTURE.md` files advertise a number of
features that are not actually implemented in code (verified by reading every adapter):

| Advertised feature | Reality in code (2026-06-19) |
|---|---|
| Homography ghost prediction (green `H-PROJ`, "the signature feature") | **Not implemented.** No `cv2.findHomography`, no foot-point collection. `generate_predictions` only emits `WORLD_PROJECTION`. |
| Pixel extrapolation ghosts (red `EXTRAP`) | **Not implemented.** Frontend has a render branch; backend never sends it. |
| Cross-camera appearance re-ID (64-dim HSV) | `compute_appearance()` exists but is **never called**; `Detection.appearance` is always `None`. Tracking is pure-IoU; cross-camera match is distance-only. |
| Sensor-trust scoring ∈ [0.1, 1.0] | Kalman `update()` takes a `sensor_trust` param but it is always the default `1.0`. No trust tracking. |
| Adaptive Kalman by confidence + bbox area + trust | Only confidence scales `R`. |
| GPS + IMU fusion | `sensor_data` messages are received then discarded (`pass`); never fused. |
| DeepSORT → Hungarian → Centroid fallback chain | Only Hungarian (+ greedy fallback). |
| Compass ribbon / threat ring HUD | Not in the canvas renderer; only corner HUD brackets are drawn. |

There is also a correctness trap: the world model is gated on `CAMERA_POSITIONS`.
With none configured, `pixel_to_world` returns `None`, so **zero world objects and zero
predictions are produced** — viewers see only raw detections and per-camera tracks.

## Definition of done (per PR)

Every PR must be green on the **comprehensive** gate, enforced in CI:

- `pytest` (backend unit) green; coverage does not regress on touched code.
- `ruff` clean (backend + live scripts; `scripts/archive/` excluded).
- `mypy` clean (`backend/app`, pragmatic config — ratchets up over time).
- `eslint` clean + `jest`/RTL green + `npm run build` succeeds (frontend).

## Decisions

- **Direction:** Both, sequenced — Phase A (honest & solid) then Phase B (features).
- **Workflow:** one branch + PR per chunk; the loop pauses at each opened PR for review.
- **`CAMERA_POSITIONS` no-op fix (A2):** auto-synthesize a default per-camera calibration
  so single-camera setups produce world objects out of the box (with a loud one-time log).
- **DeepSORT (B6):** dropped from committed scope to avoid a heavy optional dependency;
  may be revisited later.

## Phase A — Honest & Solid (tested foundation)

### A0 · Tooling & CI foundation  *(this PR)*
- Add `ruff` + `pytest-cov` config to `pyproject.toml`; add `mypy` config.
- Add frontend test infra: `@testing-library/react` + `jest-dom`, `setupTests.js`, a smoke test.
- Rewrite `.github/workflows/ci.yml`: backend job (ruff + mypy + pytest w/ coverage) and a
  new frontend job (eslint via build + jest + build).
- Fix the lint/type debt this surfaces (scoped, no behaviour change): exclude
  `scripts/archive/`, auto-fix unused imports, replace bare `except:` with `except Exception:`,
  fix the 12 mypy errors (incl. genuine bug: `PerceptionSnapshot` undefined in `ports.py`).

### A1 · Make the docs honest
- README + both `ARCHITECTURE.md`: move advertised-but-unbuilt features to a clearly-marked
  **Roadmap / "planned"** status; Phase B flips each back to "implemented" as it ships.
- Fix concrete drift: mobile handshake `target_fps` (not actually sent), `MOBILE_CAMERA_QUALITY`
  vs README naming, duplicated `CORS`/`AUTH` blocks in `.env.example`, test-count claims,
  `requires-python` vs local 3.13.

### A2 · Fix the world-model silent no-op + dead code
- Auto-default calibration when `CAMERA_POSITIONS` is empty (loud one-time log + docs).
- Mark genuinely inert paths (`appearance` cost) with `TODO(phaseB)` so they stop misleading.
- Tests: world model with **and** without `CAMERA_POSITIONS`.

### A3 · Robustness & bug hardening
- Fix the `list(self.tracks.keys())[idx]` dict-ordering coupling in tracking association.
- Audit error paths / input validation; regression test each fix.

### A4 · Coverage & type lift
- Raise backend coverage on core modules (`services`, `world_model`, `tracking`, `camera`);
  make `mypy` clean. May fold into A2/A3.

## Phase B — Make it real (each = its own PR, on the tested base)

- **B1 · Appearance re-ID** — call `compute_appearance`, populate `Detection.appearance`,
  activate the appearance term in tracking + require appearance similarity in cross-camera match.
- **B2 · Cross-camera homography (`H-PROJ`)** — foot-point correspondence collection,
  `cv2.findHomography` + RANSAC, Path-A green ghosts, reprojection-error flush.
- **B3 · Pixel extrapolation (`EXTRAP`)** — Path-B red dead-reckoning ghosts, adaptive budget.
- **B4 · GPS/IMU fusion** — consume `sensor_data`, populate `VirtualCamera` GPS/heading, fuse.
- **B5 · Sensor-trust scoring + adaptive Kalman** — trust ∈ [0.1, 1], adaptive `R` by area + trust.
- **B6 · (deferred) DeepSORT + centroid fallback chain.**

Each Phase-B PR flips its README/ARCHITECTURE status to "implemented" and adds tests proving
the path emits the right `PredictionMethod`/behaviour.

## Out of scope (for now)

- New product surfaces (compass ribbon / threat ring) unless a later phase adds them.
- Hardware-in-the-loop / Jetson integration tests (CI stays pure-Python, import-skipped).
