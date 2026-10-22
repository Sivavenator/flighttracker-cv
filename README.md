# FlightTracker CV

Real-time aircraft detection and identification using a fixed webcam, YOLOv8, and FlightRadar24.

A fixed camera points at the sky. Every aircraft that crosses the frame is:
1. **Detected** by a YOLOv8m model trained on FGVC-Aircraft (100 classes, mAP50 = 0.96)
2. **Matched to a live FR24 flight** via bearing — pixel X position maps to a compass bearing, which is matched against nearby FR24 flights
3. **Displayed** with callsign, airline, origin/destination, and altitude on a live HUD

---

## Results

| Metric | Value |
|--------|-------|
| mAP50 | 0.960 |
| mAP50-95 | 0.944 |
| Best class | 707-320 (0.995) |
| Worst class | C-47 (0.558) |
| Training epochs | 100 |
| Dataset | FGVC-Aircraft 2013b (10,000 images, 100 classes) |

---

## Quick Start (Docker)

### Prerequisites
- Docker Desktop with GPU support
- NVIDIA Container Toolkit
- Trained weights (download from [Releases](../../releases))

### Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/flighttracker-cv.git
cd flighttracker-cv

# 2. Place weights
mkdir -p weights
cp /path/to/best.pt weights/best.pt

# 3. Launch web dashboard
docker compose up

# 4. Open browser
# http://localhost:5000
```

### Test with a video file

```bash
docker compose run --rm flighttracker \
  --source /app/test_videos/test_video.mp4 \
  --no-window --web
```

---

## Quick Start (Local)

```bash
# Python 3.11 required
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Run with webcam
python main.py --source 0

# Run with video file
python main.py --source test_videos/test_video.mp4

# Run headless with web dashboard
python main.py --no-window --web
```

---

## Configuration

Edit `config.yaml` before running:

```yaml
camera:
  heading_deg: 67.0      # compass direction camera points (0=N, 90=E)
  hfov_deg: 60.0         # horizontal field of view of your lens
  location:
    lat: 1.337771        # camera GPS coordinates
    lon: 103.967680

models:
  detector:
    weights_path: weights/best.pt
```

---

## How bearing-first FR24 matching works

The camera is fixed. Pixel X position in the frame maps linearly to a compass bearing:

```
bearing = camera_heading + (pixel_x - frame_width/2) * (hfov / frame_width)
```

FR24 returns all flights within the search radius. The system matches each detection to the FR24 flight whose bearing from the camera location is closest, then validates with altitude and speed plausibility. No livery recognition or classifier needed for identification.

---

## Project Structure

```
flight_tracker_cv/
├── main.py                  # Entry point
├── config.yaml              # Camera, model, FR24 settings
├── models/
│   ├── detector.py          # YOLOv8 + ByteTrack wrapper
│   ├── classifier.py        # EfficientNetV2 classifier (optional)
│   └── flight_tracker.py    # FR24 / OpenSky adapter
├── utils/
│   ├── overlay.py           # HUD rendering
│   ├── geo_utils.py         # Bearing math
│   ├── logger.py            # SQLite detection log
│   └── sound.py             # TTS + audio alerts
├── training/
│   └── train_detector.py    # YOLOv8 fine-tuning script
├── web/
│   └── dashboard.py         # Flask web dashboard
└── data/
    └── aircraft_detection/  # Dataset (not in repo)
```

---

## Training your own detector

```bash
# Prepare FGVC-Aircraft dataset
python main.py --prepare-dataset

# Train (requires GPU)
python main.py --train-detector
```

Trained weights are saved to `runs/train/aircraft_detector-*/weights/best.pt`.

---

## License

MIT
