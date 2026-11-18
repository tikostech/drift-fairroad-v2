# SL gate demo


Checklist:

- No other FL run is active (the demo refuses to start if port 5544 is busy).
- The services are up and the CBR service serves `case_base_v2` (the deployed v6 case base). The demo checks
  this first and stops with a clear message if it can't reach it.
- The terminal font is large enough for the room.

## Running it live

```
.venv-gpu/Scripts/python demo_sl_gate.py --pause
```

`--pause` waits for Enter before each step so you can talk. If you are short of time, skip training and
present the last runs instead:

```
.venv-gpu/Scripts/python demo_sl_gate.py --reuse
```

