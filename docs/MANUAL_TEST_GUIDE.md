# Manual Test Guide — Enroll → Lock → Track

Step-by-step instructions to enroll a **new person**, **lock** onto them manually, and **track** them with the pan camera. All commands are for **Windows PowerShell** from the project root.

**Related:** [ASSESSMENT_TEST_GUIDE.md](ASSESSMENT_TEST_GUIDE.md) (full assessment demo) · [BENAX_INTEGRATED_SYSTEM.md](BENAX_INTEGRATED_SYSTEM.md) (architecture)

---

## Before you start

```powershell
cd c:\Users\RCA\Documents\FaceLocking-main\FaceLocking-main
.\venv\Scripts\Activate.ps1
```

If `python` is not found, use the venv directly (no activation needed):

```powershell
.\venv\Scripts\python.exe -m src.enroll --help
```

| Item | Typical value |
|------|----------------|
| Pan USB camera | index **1** |
| Camera rotation | **180** (upside-down mount) |
| Speaker name | e.g. **Faith** or **Alex** (use the same name everywhere) |
| ESP8266 | Flashed with `vision/faithlock/movement` topic |
| MQTT broker | `157.173.101.159:1883` |

Find your camera index if needed:

```powershell
python scripts\probe_cameras.py
```

---

## Step 1 — Enroll a new person

Enrollment saves face photos and builds one recognition template in `data/db/face_db.npz`.

### Option A: Fresh enroll (recommended for a new person)

Clears old crops for that name and starts clean.

```powershell
python -m src.enroll --name Alex --camera-index 1 --camera-rotate 180 --fresh
```

Replace `Alex` with the person’s name.

### Option B: Clear entire database first (only one person in system)

```powershell
python scripts\reset_speaker_db.py
python -m src.enroll --name Alex --camera-index 1 --camera-rotate 180 --fresh
```

### In the enrollment window

1. **Only the person being enrolled** should be in frame (pan camera).
2. Turn head slightly left/right between shots; neutral and smiling helps.
3. Press keys in the enroll window:

| Key | Action |
|-----|--------|
| **SPACE** | Save one face photo (when ready) |
| **S** | Finish enrollment (need **8+** photos; **12+** recommended) |
| **R** | Discard new photos (keeps old crops on disk) |
| **Q** | Quit without saving |

### Verify enrollment worked

```powershell
# Should list your name in the database metadata
type data\db\face_db.json

# Should show saved face crops
dir data\enroll\Alex\*.jpg
```

---

## Step 2 — Start dashboard (optional but recommended)

Open a **new** terminal:

```powershell
cd c:\Users\RCA\Documents\FaceLocking-main\FaceLocking-main
.\venv\Scripts\Activate.ps1
python -m http.server 8765 --directory dashboard
```

Open in browser: **http://localhost:8765/index.html**

---

## Step 3 — Start tracking

In another terminal:

```powershell
cd c:\Users\RCA\Documents\FaceLocking-main\FaceLocking-main
.\venv\Scripts\Activate.ps1
python addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Alex
```

Use the **same name** you used in enrollment (`--speaker-name Alex`).

You should see:

- Tracking window: **BENAX Speaker Tracking**
- Console: `Auto-lock speaker: OFF - press L to lock`
- Console: `Auto-unlock after: 40s without seeing locked speaker`
- Console: `[MQTT] Connected to ...` (if broker is reachable)

---

## Step 4 — Lock the person (manual)

Auto-lock is **off** by default. You must lock manually.

1. Stand in front of the **pan camera** (only the enrolled person, or select them if multiple faces).
2. Wait until the label shows the correct **name** (e.g. `Alex`) with a confidence %.
3. Press **L** in the tracking window.

| Result | What you see |
|--------|----------------|
| Locked | Orange border, “(LOCKED)”, console: `[SpeakerLock] Locked onto Alex` |
| Dashboard | “Locked on Alex”, phase **Tracking** |

### If more than one face is in frame

| Key | Action |
|-----|--------|
| **A** or **←** | Select previous face |
| **F** or **→** | Select next face |
| **L** | Lock onto selected face (must match enrolled name) |

### Unlock

Press **L** again while locked → unlocks and sends **STOPPED** to the servo.

---

## Step 5 — Track the person

After lock:

1. **Move left/right** slowly — camera should pan to keep you centered.
2. Dashboard shows **Panning left/right** or **Holding center**.
3. ESP Serial Monitor shows `[MQTT] Received: MOVED_LEFT` / `MOVED_RIGHT`.

### What the system does when you disappear

| Situation | Behavior |
|-----------|----------|
| Brief occlusion (hand, blink) | May show **OUT_OF_FRAME** in logs; lock stays |
| Leave frame completely | **SCAN** — camera sweeps to find you |
| Return while locked | Tracking resumes on same person |
| Gone **40+ seconds** | Auto-unlock + **STOPPED** (press **L** again to re-lock) |

---

## Step 6 — Stop tracking

In the tracking window, press **Q**.

Evidence is saved automatically to:

- `logs/evidence/session_YYYYMMDD_HHMMSS.csv`
- `logs/evidence/session_YYYYMMDD_HHMMSS.jsonl`

---

## Quick reference — all commands

### PowerShell (run from project root)

```powershell
# Activate environment (every new terminal)
.\venv\Scripts\Activate.ps1

# Reset all speakers (optional)
python scripts\reset_speaker_db.py

# Enroll new person (manual photos)
python -m src.enroll --name Alex --camera-index 1 --camera-rotate 180 --fresh

# Validate software
python addons\mqtt_servo_tracking\validate_system.py

# Test servo only (no face AI)
python scripts\test_servo_mqtt.py --steps 15

# Test MQTT from PC
python scripts\diagnose_mqtt.py

# Dashboard
python -m http.server 8765 --directory dashboard

# Track + lock + follow
python addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Alex
```

### Enrollment window keys

| Key | Action |
|-----|--------|
| SPACE | Save photo |
| S | Finish enrollment |
| R | Reset new photos |
| Q | Quit |

### Tracking window keys

| Key | Action |
|-----|--------|
| **L** | Lock / unlock speaker |
| **A** / **←** | Select face left |
| **F** / **→** | Select face right |
| **+** / **-** | Stricter / looser match threshold |
| **R** | Reload face database |
| **D** | Toggle debug overlay |
| **Q** | Quit |

---

## Optional flags

```powershell
# Auto-lock when enrolled person appears (no L key)
python addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Alex --auto-lock-speaker

# Change auto-unlock timeout (default 40 seconds)
python addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Alex --lock-timeout-sec 60

# Disable search sweep when person is lost (hold pan instead)
python addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Alex --no-scan
```

---

## One-page workflow (copy/paste)

Replace `Alex` with your speaker name.

```powershell
cd c:\Users\RCA\Documents\FaceLocking-main\FaceLocking-main
.\venv\Scripts\Activate.ps1

# 1. Enroll
python -m src.enroll --name Alex --camera-index 1 --camera-rotate 180 --fresh
#    SPACE = photos, S = save, Q = quit

# 2. Dashboard (new terminal)
python -m http.server 8765 --directory dashboard

# 3. Track (new terminal)
python addons\mqtt_servo_tracking\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Alex
#    L = lock, move to test follow, Q = quit
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera wrong / upside down | `--camera-index 1 --camera-rotate 180` |
| “Speaker not in database” | Enroll with same `--name` as `--speaker-name` |
| Servo not moving | Re-flash ESP; run `test_servo_mqtt.py` |
| Everyone gets same label | Re-enroll with only one person in frame; use `--fresh` |
| **Hyguette (or new person) not followed** | DB had multiple names — use `--speaker-name Hyguette`; matcher now uses only that template. If label shows **Unknown**, press **-** to loosen threshold or add enroll photos |
| Dashboard empty | Open `http://localhost:8765/index.html` (not file://) |
