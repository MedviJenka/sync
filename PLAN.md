# Drone Gesture Control Improvement Plan

## Current baseline

The project already detects faces and hands through OpenCV/CVZone, draws a low-clutter overlay, and maps index-finger direction to drone `takeoff`/`land` commands through PySimVerse.

Observed baseline issue: `uv run pytest -q` does not collect tests because the project uses a `src/` package layout while production modules still live at repository root (`main.py`, `vision/`).

## Target architecture

```text
Camera frame
  -> Hand detector
  -> Gesture recognizer
  -> Temporal stabilizer
  -> Safety command policy
  -> Drone adapter
  -> Overlay status/logging
```

## Phase 0 — Package and test foundation

- Move runtime code under the installable `drone` package.
- Use package imports (`drone.vision...`) instead of root-level `vision`/`main` imports.
- Keep the console script as `drone = "drone:main"`.
- Make `uv run pytest -q` collect and run from a clean checkout.

## Phase 1 — Safety before richer flight controls

- Require gesture stability across consecutive frames before executing a drone command.
- Add command cooldowns so one physical gesture cannot spam commands.
- Keep command execution idempotent against drone state (`takeoff` only when grounded, `land` only when flying).
- Preserve a no-drone mode for camera/overlay development.

## Phase 2 — Explicit arming and emergency control

- Add an `armed` state; ignore flight commands while disarmed.
- Add a deliberate arm/disarm gesture.
- Add emergency land by gesture and keyboard.
- Auto-land on camera failure, drone disconnect, or prolonged hand-tracking loss.

## Phase 3 — Development loop without hardware

- Add a simulator drone adapter implementing the same protocol as PySimVerse.
- Add recorded landmark replay fixtures.
- Emit command timeline logs with frame number, stable gesture duration, command, and decision reason.

## Phase 4 — Richer control and UX

- Show command state on overlay: `DISARMED`, `ARMED`, `HOLD`, `READY`, `COMMAND SENT`, `COOLDOWN`, `EMERGENCY LAND`.
- Add gestures for hover/stop, yaw, and follow-hand mode only after safety gates are proven.
- Add calibration for camera angle, user distance, and gesture thresholds.

## Implementation status

### Completed

1. Phase 0 package foundation: runtime code now lives under `src/drone/`, imports use `drone.vision...`, and the `drone` console entrypoint resolves through `drone.cli`.
2. Phase 1 safety policy: `DroneGestureController` now requires consecutive stable detections before command execution and enforces a post-command detection cooldown.
3. Phase 1 configuration: CLI exposes `--stable-detections` and `--command-cooldown-detections`; live camera wiring passes those settings into the drone controller.
4. Phase 1 verification: focused P1 tests and the full project suite pass.
5. Phase 2 arming: gesture flight commands can run through an armed/disarmed state; the CLI starts armed for immediate demos and `--start-disarmed` restores the deliberate pinch-to-arm safety gate.
6. Phase 2 emergency landing: open palm, keyboard `E`, camera read failure, drone-controller failure, and prolonged hand-tracking loss route to emergency landing.
7. Phase 2 configuration: CLI exposes `--start-armed`, `--start-disarmed`, and `--lost-tracking-land-detections`.
8. Phase 2 verification: focused P2 tests and the full project suite pass.
