# GaitGuard — macbook-demo

Quickstart (macOS)

1. Clone and checkout demo branch:
   git fetch origin
   git checkout macbook-demo

2. Create & activate venv:
   python3 -m venv .venv
   source .venv/bin/activate

3. Install deps:
   pip install -r requirements.txt
   pip install certifi  # helps mediapipe model download on macOS

4. Ensure directories exist and create .env if needed:
   mkdir -p notebooks docs logs
   cp .env.example .env || true

5. Run the demo:
   python src/demos/demo_camera.py --fps 5

Notes:
- The demo uses MediaPipe Pose and will download the TFLite model on first run.
- Baselines use a warm-up of 10 embeddings per new track before scoring begins.
- Timeseries rows are buffered and flushed to SQLite every 10s by default.
- To quit the demo window press `q` or send SIGINT/Ctrl+C to flush and exit cleanly.
