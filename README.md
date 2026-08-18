# Drone Gesture Control

Run the live camera overlay:

```bash
uv run drone
```

Drone commands start armed by default, so after two stable detections the command above accepts an index finger pointed up as `takeoff` and an index finger pointed down as `land`.

For safer sessions where takeoff/land should be ignored until you deliberately arm them, start disarmed explicitly:

```bash
uv run drone --start-disarmed
```

When disarmed, the overlay shows `DISARMED - PINCH TO ARM`; pinch your thumb and index finger to arm gesture flight commands.

Emergency land triggers:

- open palm gesture
- keyboard `E`
- camera read failure
- prolonged hand-tracking loss
