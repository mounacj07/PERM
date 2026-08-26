# PERM — Predictive Edge Resource Management

PERM is a lightweight edge-monitoring system that watches CPU, memory,
and response latency on a machine in real time, and predicts what
those numbers will look like 30 seconds from now using a Random
Forest model. The goal is to catch resource problems before they
happen instead of just reacting to them after the fact.

Everything runs locally — no external ML APIs or cloud dependencies.

## How it works

A Flask service (`telemetry_collector.py`) exposes a workload endpoint
and samples CPU/memory usage every second in the background. We use
Locust (`locustfile.py`) to simulate realistic traffic — steady load,
gradual ramps, and sudden bursts — against that endpoint to generate
training data. From there, a feature engineering step builds
30-second sliding window features, a Random Forest model is trained
on it, and an inference service uses that model to predict load 30
seconds ahead. If predictions cross defined SLO thresholds, an alert
engine raises warnings before the system actually becomes overloaded,
and a Matplotlib dashboard shows all of this live.


## Setup

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd perm
pip install -r requirements.txt
```

A virtual environment is recommended but not required:

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
```

## Running it

**1. Start the telemetry collector** (keep this running in its own terminal):

```bash
python telemetry_collector.py --port 5000
```

This starts the Flask app and a background thread that samples system
metrics every second. Data gets written to `telemetry_log.csv`.

A few useful endpoints while it's running:
- `GET /health` — check it's alive
- `GET /metrics/latest` — most recent telemetry sample
- `GET /metrics/history?seconds=60` — recent history
- `POST /api/process` — the workload endpoint Locust drives

**2. Generate traffic for training data** (separate terminal):

```bash
locust -f locustfile.py --host http://127.0.0.1:5000 --headless
```

This cycles through steady, ramp, and burst traffic patterns
repeatedly (5 cycles by default, about 41 minutes) and logs each
phase transition to `phase_log.csv` so we know which traffic profile
each row of telemetry belongs to.

For a quicker test run instead of the full 41 minutes:

```bash
# PowerShell
$env:PERM_LOCUST_CYCLES = 1
locust -f locustfile.py --host http://127.0.0.1:5000 --headless
```

Runs are cumulative — every time you run Locust it adds to the
existing CSVs, so the dataset can be built up across multiple
sessions instead of needing one long sitting.

**3. Check what got collected:**

```bash
# PowerShell
Import-Csv telemetry_log.csv | Format-Table -AutoSize

# bash
column -s, -t telemetry_log.csv | less -S
```

