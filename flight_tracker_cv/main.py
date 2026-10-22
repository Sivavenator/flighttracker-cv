"""
FlightTracker CV — Main Application Entry Point
Real-time aircraft detection, classification, and flight data correlation.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

# Rich for pretty console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("flighttracker")

# ---------------------------------------------------------------------------
# Lazy imports — handled gracefully if dependencies are missing
# ---------------------------------------------------------------------------

def _import_detector():
    from models.detector import AircraftDetector
    return AircraftDetector

def _import_classifier():
    from models.classifier import AircraftClassifier
    return AircraftClassifier

def _import_flight_tracker():
    from models.flight_tracker import FlightTrackerFactory
    return FlightTrackerFactory

def _import_hud():
    from utils.overlay import HUDRenderer
    return HUDRenderer

def _import_logger():
    from utils.logger import FlightLogger
    return FlightLogger

def _import_sound():
    from utils.sound import SoundManager
    return SoundManager

def _import_screenshot():
    from utils.screenshot import ScreenshotManager
    return ScreenshotManager

def _import_geo():
    from utils.geo_utils import is_daytime, bearing_from_image_position
    return is_daytime, bearing_from_image_position

def _import_dashboard():
    from web.dashboard import run_dashboard
    return run_dashboard


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config with sensible defaults."""
    defaults: dict = {
        "app": {"name": "FlightTracker CV", "version": "1.0.0", "debug": False},
        "camera": {"device": 0, "width": 1920, "height": 1080, "fps": 30},
        "models": {
            "detector": {"weights": "yolov8m.pt", "confidence": 0.5, "iou_threshold": 0.45},
            "classifier": {"weights": None, "num_aircraft_classes": 100, "num_airline_classes": 60, "top_k": 3},
        },
        "flight_tracker": {
            "provider": "flightradar24",
            "update_interval_s": 5,
            "radius_km": 50,
            "camera_lat": 1.3521,
            "camera_lon": 103.8198,
        },
        "overlay": {"show_trails": True, "hud_enabled": True, "night_mode_auto": True},
        "logging": {"db_path": "logs/detections.db", "level": "INFO"},
        "notifications": {"sound": True, "tts": True, "screenshot": True},
        "web": {"host": "0.0.0.0", "port": 5000, "frame_quality": 70},
        "data": {"screenshots_dir": "screenshots"},
    }

    try:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep merge: user overrides defaults
        def _merge(base: dict, override: dict) -> dict:
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k] = _merge(base[k], v)
                else:
                    base[k] = v
            return base
        return _merge(defaults, user_cfg)
    except FileNotFoundError:
        logger.warning("Config file %s not found, using defaults.", config_path)
        return defaults
    except yaml.YAMLError as exc:
        logger.error("YAML parse error in %s: %s", config_path, exc)
        return defaults


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class FlightTrackerApp:
    """
    Orchestrates the full detection pipeline:
      1. Camera capture
      2. YOLOv8 detection + ByteTrack multi-object tracking
      3. EfficientNetV2-M classification (type + airline)
      4. FlightRadar24 / OpenSky flight data matching
      5. HUD overlay rendering
      6. Web dashboard streaming
      7. Sound / TTS notifications
      8. Screenshot capture and SQLite logging
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

        # Queues for web dashboard
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.detection_queue: queue.Queue = queue.Queue(maxsize=100)

        # Shared mutable config (updated via /api/config)
        self.runtime_cfg: dict = {
            "sound_enabled": config.get("notifications", {}).get("sound", True),
            "tts_enabled": config.get("notifications", {}).get("tts", True),
            "screenshot_enabled": config.get("notifications", {}).get("screenshot", True),
            "show_trails": config.get("overlay", {}).get("show_trails", True),
            "confidence_threshold": config.get("models", {}).get("detector", {}).get("confidence", 0.5),
        }

        # Components — initialised in setup()
        self.detector = None
        self.classifier = None
        self.flight_tracker = None
        self.hud = None
        self.db_logger = None
        self.sound = None
        self.screenshot_mgr = None

        # Camera geometry — used for bearing-first FR24 matching
        cam_cfg = config.get("camera", {})
        self.camera_heading = float(cam_cfg.get("heading_deg", 0.0))
        self.camera_hfov    = float(cam_cfg.get("hfov_deg", 60.0))
        self.camera_vfov    = float(cam_cfg.get("vfov_deg", 34.0))

        # Per-frame state
        self._last_fr24_flights: list = []
        self._matched_flights: Dict[int, Any] = {}   # track_id -> MatchResult
        self._frame_count = 0
        self._fps_counter_ts = time.time()
        self._fps = 0.0
        self._panel_info: dict = {}   # latest detection for info panel

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise all components. Non-fatal errors are logged and skipped."""
        self._setup_logging()
        logger.info("Initialising FlightTracker CV...")

        self._setup_detector()
        self._setup_classifier()
        self._setup_flight_tracker()
        self._setup_hud()
        self._setup_db_logger()
        self._setup_sound()
        self._setup_screenshot()

    def _setup_logging(self) -> None:
        level_name = self.cfg.get("logging", {}).get("level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    def _setup_detector(self) -> None:
        try:
            AircraftDetector = _import_detector()
            det_cfg = self.cfg.get("models", {}).get("detector", {})
            weights = det_cfg.get("weights_path") or det_cfg.get("weights", "yolov8m.pt")
            # Prefer fine-tuned weights if they exist
            ft_path = ROOT / "models" / "weights" / "aircraft_detector_best.pt"
            if ft_path.exists():
                weights = str(ft_path)
                logger.info("Using fine-tuned detector weights: %s", weights)
            self.detector = AircraftDetector(
                weights_path=weights,
                confidence=self.runtime_cfg["confidence_threshold"],
                iou_threshold=det_cfg.get("iou_threshold", 0.45),
                use_tracker=True,
            )
            logger.info("Detector ready: %s", weights)
        except Exception as exc:
            logger.error("Detector init failed: %s", exc)
            self.detector = None

    def _setup_classifier(self) -> None:
        try:
            AircraftClassifier = _import_classifier()
            cls_cfg = self.cfg.get("models", {}).get("classifier", {})
            self.classifier = AircraftClassifier(
                model_path=cls_cfg.get("weights"),
                num_aircraft_classes=cls_cfg.get("num_aircraft_classes", 100),
                num_airline_classes=cls_cfg.get("num_airline_classes", 60),
            )
            logger.info("Classifier ready.")
        except Exception as exc:
            logger.error("Classifier init failed: %s", exc)
            self.classifier = None

    def _setup_flight_tracker(self) -> None:
        try:
            FlightTrackerFactory = _import_flight_tracker()
            ft_cfg = self.cfg.get("flight_tracker", {})
            cam_loc = self.cfg.get("camera", {}).get("location", {})
            self.flight_tracker = FlightTrackerFactory.create(
                provider=ft_cfg.get("provider", "flightradar24"),
                lat=cam_loc.get("lat", 1.337771),
                lon=cam_loc.get("lon", 103.967680),
                radius_km=ft_cfg.get("search_radius_km", ft_cfg.get("radius_km", 50)),
                update_interval_s=ft_cfg.get("update_interval", ft_cfg.get("update_interval_s", 5)),
            )
            self.flight_tracker.start()
            logger.info("Flight tracker started (%s).", ft_cfg.get("provider", "flightradar24"))
        except Exception as exc:
            logger.warning("Flight tracker unavailable: %s", exc)
            self.flight_tracker = None
        

    def _setup_hud(self) -> None:
        try:
            HUDRenderer = _import_hud()
            self.hud = HUDRenderer()
            logger.info("HUD renderer ready.")
        except Exception as exc:
            logger.error("HUD init failed: %s", exc)
            self.hud = None

    def _setup_db_logger(self) -> None:
        try:
            FlightLogger = _import_logger()
            db_path = ROOT / self.cfg.get("logging", {}).get("db_path", "logs/detections.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_logger = FlightLogger(str(db_path))
            logger.info("DB logger ready at %s", db_path)
        except Exception as exc:
            logger.error("DB logger init failed: %s", exc)
            self.db_logger = None

    def _setup_sound(self) -> None:
        try:
            SoundManager = _import_sound()
            self.sound = SoundManager(
                sound_enabled=self.runtime_cfg["sound_enabled"],
                tts_enabled=self.runtime_cfg["tts_enabled"],
            )
            logger.info("Sound manager ready.")
        except Exception as exc:
            logger.warning("Sound manager unavailable: %s", exc)
            self.sound = None

    def _setup_screenshot(self) -> None:
        try:
            ScreenshotManager = _import_screenshot()
            ss_dir = ROOT / self.cfg.get("data", {}).get("screenshots_dir", "screenshots")
            ss_dir.mkdir(parents=True, exist_ok=True)
            self.screenshot_mgr = ScreenshotManager(output_dir=str(ss_dir))
            logger.info("Screenshot manager ready.")
        except Exception as exc:
            logger.warning("Screenshot manager unavailable: %s", exc)
            self.screenshot_mgr = None

    # ------------------------------------------------------------------
    # Camera open
    # ------------------------------------------------------------------

    def _open_camera(self, source) -> cv2.VideoCapture:
        """Open camera or video file. Returns VideoCapture."""
        # Convert numeric string to int
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        cam_cfg = self.cfg.get("camera", {})
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width", 1920))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 1080))
            cap.set(cv2.CAP_PROP_FPS,          cam_cfg.get("fps", 30))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info("Opened %s at %dx%d @ %.1f fps", source, actual_w, actual_h, actual_fps)
        return cap

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, source=0, show_window: bool = True, web: bool = False,
            skip_frames: int = 0) -> None:
        """
        Main processing loop.

        source:       camera device index (int), path to video file (str), or RTSP URL
        show_window:  display OpenCV window
        web:          start Flask dashboard in background thread
        skip_frames:  process every Nth frame (0 = all frames)
        """
        self._running = True

        # Start web dashboard
        if web:
            self._start_dashboard()

        # Night mode auto-detection helpers
        try:
            is_daytime_fn, bearing_fn = _import_geo()
        except Exception:
            is_daytime_fn = None
            bearing_fn = None

        ft_cfg = self.cfg.get("flight_tracker", {})
        cam_lat = ft_cfg.get("camera_lat", 1.3521)
        cam_lon = ft_cfg.get("camera_lon", 103.8198)

        try:
            cap = self._open_camera(source)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return

        self._cap = cap
        window_name = "FlightTracker CV — Press Q to quit"

        frame_idx = 0
        process_every = max(1, skip_frames + 1)

        logger.info("Starting main loop. Press Q or Ctrl+C to stop.")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    # End of video file — loop or exit
                    if isinstance(source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Camera read failed.")
                        time.sleep(0.05)
                        continue

                frame_idx += 1
                self._frame_count += 1

                # FPS calculation
                now = time.time()
                elapsed = now - self._fps_counter_ts
                if elapsed >= 2.0:
                    self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_counter_ts = now

                # Night mode
                if is_daytime_fn and self.cfg.get("overlay", {}).get("night_mode_auto", True):
                    import datetime
                    day = is_daytime_fn(cam_lat, cam_lon, datetime.datetime.utcnow())
                    if self.hud:
                        self.hud.set_night_mode(not day)

                # Process only every Nth frame for performance
                if frame_idx % process_every != 0:
                    annotated = frame.copy()
                else:
                    annotated = self._process_frame(frame, frame_idx)

                # HUD overlay (always drawn)
                if self.hud:
                    fr24_ok = (self.flight_tracker is not None)
                    annotated = self.hud.draw_hud(
    annotated,
    fps=self._fps,
    detection_count=len(self._matched_flights),
    location_name="Singapore",
    fr24_connected=fr24_ok,
    bearing=self.camera_heading,
)
                # Bottom-left aircraft info panel
                if self.hud and self._panel_info:
                    annotated = self.hud.draw_aircraft_panel(
                        annotated,
                        aircraft_type=self._panel_info.get("aircraft_type", "Unknown"),
                        airline=self._panel_info.get("airline", "Unknown"),
                        origin=self._panel_info.get("origin", "Unknown"),
                        destination=self._panel_info.get("destination", "Unknown"),
                        accuracy=self._panel_info.get("accuracy", 0.0),
                    )

                # Push to web dashboard
                self._push_frame(annotated)

                # OpenCV window
                if show_window:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        logger.info("User quit.")
                        break
                    elif key == ord('n'):
                        if self.hud:
                            self.hud.set_night_mode(True)
                    elif key == ord('d'):
                        if self.hud:
                            self.hud.set_night_mode(False)
                    elif key == ord('s'):
                        self.runtime_cfg["sound_enabled"] = not self.runtime_cfg["sound_enabled"]
                        if self.sound:
                            self.sound.toggle_sound(self.runtime_cfg["sound_enabled"])

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.shutdown()
            if show_window:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Run detection + classification + FR24 matching on one frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # --- Detection ---
        detections = []
        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                logger.debug("Detection error: %s", exc)

        # --- Classification + FR24 matching per detection ---
        nearby_flights = []
        if self.flight_tracker is not None:
            try:
                nearby_flights = self.flight_tracker.get_nearby_flights()
            except Exception:
                pass

        new_matched: Dict[int, Any] = {}

        for det in detections:
            track_id = det.track_id if det.track_id is not None else -1
            bbox = det.bbox_xyxy  # [x1, y1, x2, y2]

            # Crop for classifier
            classification = None
            if self.classifier is not None:
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        classification = self.classifier.classify(crop)
                except Exception as exc:
                    logger.debug("Classification error: %s", exc)

            # Estimate distance & bearing
            distance_km = None
            bearing_deg = None
            if self.detector is not None:
                try:
                    distance_km = self.detector.estimate_distance(
                        bbox,
                        known_wingspan_m=34.0,   # A320-class default
                        focal_length_px=getattr(self.detector, "focal_length_px", 1200),
                        frame_width=w,
                    )
                except Exception:
                    pass
                try:
                    hfov = getattr(self.detector, "hfov_degrees", 60.0)
                    bearing_deg = self.detector.get_bearing_from_bbox(
                        bbox, w, h, hfov_degrees=hfov, camera_heading_deg=self.camera_heading
                    )
                except Exception:
                    pass

            # FR24 matching
            matched_flight = None
            if nearby_flights and bearing_deg is not None:
                ac_type = (classification.aircraft_type if classification else None) or det.class_name
                try:
                    matched_flight = self.flight_tracker.match_to_detection(
                        bearing=bearing_deg,
                        distance=distance_km or 5.0,
                        detected_type=ac_type,
                    )
                except Exception as exc:
                    logger.debug("FR24 match error: %s", exc)

            if track_id >= 0 and matched_flight:
                new_matched[track_id] = matched_flight

            # Draw bounding box
            if self.hud is not None:
                label = det.class_name
                if classification and classification.aircraft_type and classification.aircraft_type != "unknown":
                    label = classification.aircraft_type
                colour = self.detector.BOX_COLOURS[track_id % len(self.detector.BOX_COLOURS)] \
                    if self.detector else (0, 229, 255)
                annotated = self.hud.draw_bounding_box(
                    annotated, bbox, label, det.confidence, colour, track_id
                )
                if matched_flight:
                    annotated = self.hud.draw_flight_info(
                        annotated, bbox, matched_flight, classification, distance_km
                    )
                if self.runtime_cfg.get("show_trails") and self.detector and track_id >= 0:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    self.detector.track_history[track_id].append((cx, cy))
                    annotated = self.hud.draw_trail(annotated, list(self.detector.track_history.get(track_id, [])), colour)
                # Flash on new match
                if matched_flight and track_id not in self._matched_flights:
                    annotated = self.hud.draw_trail(annotated, list(self.detector.track_history.get(track_id, [])), colour)

            # Notifications on new match
            if matched_flight and track_id not in self._matched_flights:
                self._on_new_match(matched_flight, classification, frame, bbox, bearing_deg, distance_km)

            # Log every detection (regardless of match)
            elif track_id not in self._matched_flights:
                self._push_detection_event(det, classification, matched_flight)

            # Update panel with best available info
            if detections:
                best = detections[0]
                mf = new_matched.get(best.track_id if best.track_id else -1)
                cls = None  # classification already handled above per-det
                self._panel_info = {
                    "aircraft_type": (getattr(mf, "aircraft_type", None) or best.class_name or "Unknown"),
                    "airline":       getattr(mf, "airline_name", "Unknown") or "Unknown",
                    "origin":        getattr(mf, "origin_iata",  "Unknown") or "Unknown",
                    "destination":   getattr(mf, "dest_iata",    "Unknown") or "Unknown",
                    "accuracy":      round(best.confidence * 100, 1),
                }

        self._matched_flights = new_matched
        return annotated

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_new_match(self, flight, classification, frame, bbox, bearing_deg, distance_km):
        """Handle a newly matched flight: sound, TTS, screenshot, DB, queue."""
        # Sound
        if self.sound and self.runtime_cfg.get("sound_enabled"):
            try:
                self.sound.play_matched_chime()
            except Exception:
                pass

        # TTS
        if self.sound and self.runtime_cfg.get("tts_enabled"):
            try:
                ac_type = (classification.aircraft_type if classification else None) or "Unknown"
                airline = getattr(flight, "airline_name", "") or ""
                origin = getattr(flight, "origin_iata", "") or ""
                dest = getattr(flight, "dest_iata", "") or ""
                self.sound.announce_flight(ac_type, airline, origin, dest)
            except Exception:
                pass

        # Screenshot
        ss_path = None
        if self.screenshot_mgr and self.runtime_cfg.get("screenshot_enabled"):
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop_with_context = frame[
                    max(0, y1 - 20):min(frame.shape[0], y2 + 20),
                    max(0, x1 - 20):min(frame.shape[1], x2 + 20),
                ]
                ss_path = self.screenshot_mgr.capture(crop_with_context, flight, classification)
            except Exception as exc:
                logger.debug("Screenshot error: %s", exc)

        # DB log
        if self.db_logger:
            try:
                self.db_logger.log_detection(
                    flight_data=flight,
                    classification_result=classification,
                    distance_km=distance_km,
                    bearing_deg=bearing_deg,
                    screenshot_path=ss_path,
                )
            except Exception as exc:
                logger.debug("DB log error: %s", exc)

        # Web event
        self._push_matched_event(flight, classification, distance_km)

    def _push_detection_event(self, det, classification, flight):
        event = {
            "_type": "detection",
            "aircraft_type": (classification.aircraft_type if classification else None) or det.class_name,
            "confidence": float(det.confidence),
            "track_id": det.track_id,
            "active_count": len(self._matched_flights) + 1,
        }
        if classification:
            event["airline"] = classification.airline
            event["variant"] = classification.variant
        try:
            self.detection_queue.put_nowait(event)
        except queue.Full:
            pass

    def _push_matched_event(self, flight, classification, distance_km):
        event = {
            "_type": "flight_matched",
            "flight_id": getattr(flight, "flight_id", ""),
            "callsign": getattr(flight, "callsign", ""),
            "aircraft_type": (classification.aircraft_type if classification else None) or getattr(flight, "aircraft_type", ""),
            "airline": getattr(flight, "airline_name", ""),
            "airline_name": getattr(flight, "airline_name", ""),
            "origin_iata": getattr(flight, "origin_iata", ""),
            "dest_iata": getattr(flight, "dest_iata", ""),
            "altitude_ft": getattr(flight, "altitude_ft", 0),
            "speed_kts": getattr(flight, "speed_kts", 0),
            "heading_deg": getattr(flight, "heading_deg", 0),
            "lat": getattr(flight, "lat", None),
            "lon": getattr(flight, "lon", None),
            "distance_km": distance_km,
            "variant": (classification.variant if classification else None),
        }
        try:
            self.detection_queue.put_nowait(event)
        except queue.Full:
            pass

    def _push_frame(self, frame: np.ndarray) -> None:
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except (queue.Full, queue.Empty):
                pass

    # ------------------------------------------------------------------
    # Web dashboard
    # ------------------------------------------------------------------

    def _start_dashboard(self) -> None:
        try:
            run_dashboard_fn = _import_dashboard()
            web_cfg = self.cfg.get("web", {})
            t = threading.Thread(
                target=run_dashboard_fn,
                kwargs={
                    "frame_queue": self.frame_queue,
                    "detection_queue": self.detection_queue,
                    "flight_tracker": self.flight_tracker,
                    "db_logger": self.db_logger,
                    "host": web_cfg.get("host", "0.0.0.0"),
                    "port": web_cfg.get("port", 5000),
                    "config": {**self.runtime_cfg, **web_cfg},
                },
                daemon=True,
                name="dashboard",
            )
            t.start()
            logger.info(
                "Web dashboard started at http://%s:%d",
                web_cfg.get("host", "0.0.0.0"),
                web_cfg.get("port", 5000),
            )
        except Exception as exc:
            logger.error("Dashboard start failed: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down...")
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.flight_tracker is not None:
            try:
                self.flight_tracker.stop()
            except Exception:
                pass
        if self.sound is not None:
            try:
                self.sound.shutdown()
            except Exception:
                pass
        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        if not _RICH:
            print("FlightTracker CV session summary:")
            if self.db_logger:
                stats = self.db_logger.get_stats()
                print(f"  Today: {stats.get('today_count', 0)} detections")
                print(f"  Total: {stats.get('total_detections', 0)}")
            return

        console = Console()
        table = Table(title="FlightTracker CV — Session Summary", border_style="cyan")
        table.add_column("Component", style="bold")
        table.add_column("Status")

        table.add_row("Detector",  "[green]OK[/]" if self.detector   else "[red]N/A[/]")
        table.add_row("Classifier","[green]OK[/]" if self.classifier  else "[yellow]No weights[/]")
        table.add_row("FR24",      "[green]OK[/]" if self.flight_tracker else "[yellow]Offline[/]")
        table.add_row("DB Logger", "[green]OK[/]" if self.db_logger   else "[red]N/A[/]")
        table.add_row("Sound",     "[green]OK[/]" if self.sound       else "[yellow]N/A[/]")

        if self.db_logger:
            try:
                stats = self.db_logger.get_stats()
                table.add_row("Today detections",  str(stats.get("today_count", 0)))
                table.add_row("Total detections",  str(stats.get("total_detections", 0)))
                table.add_row("Airlines seen",     str(stats.get("airlines_seen", 0)))
                table.add_row("Types seen",        str(stats.get("types_seen", 0)))
            except Exception:
                pass

        console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlightTracker CV — Real-time aircraft detection and identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Webcam, device 0
  python main.py --source 1              # Webcam, device 1
  python main.py --source video.mp4      # Video file
  python main.py --web                   # Enable web dashboard
  python main.py --no-window --web       # Headless + web
  python main.py --test                  # Run video test suite
  python main.py --train-detector        # Train YOLOv8 detector
  python main.py --train-classifier      # Train EfficientNet classifier
  python main.py --generate-report       # Generate accuracy report
        """,
    )
    parser.add_argument("--config",      default="config.yaml",     help="Path to YAML config file")
    parser.add_argument("--source",      default="0",                help="Camera device index or video file path")
    parser.add_argument("--web",         action="store_true",         help="Enable Flask web dashboard")
    parser.add_argument("--no-window",   action="store_true",         help="Suppress OpenCV display window")
    parser.add_argument("--skip-frames", type=int, default=0,         help="Process every N+1 frames")
    parser.add_argument("--test",        action="store_true",         help="Run test video suite")
    parser.add_argument("--train-detector",   action="store_true",    help="Train YOLOv8 detector")
    parser.add_argument("--train-classifier", action="store_true",    help="Train EfficientNet classifier")
    parser.add_argument("--generate-report",  action="store_true",    help="Generate HTML+PDF report")
    parser.add_argument("--export-detector",  action="store_true",    help="Export detector to ONNX")
    parser.add_argument("--prepare-dataset",  action="store_true",    help="Download and prepare FGVC dataset")
    parser.add_argument("--cpu",         action="store_true",         help="Force CPU inference")
    return parser.parse_args()


def _handle_train_detector(cfg: dict) -> None:
    from training.train_detector import main as train_main
    train_main()


def _handle_train_classifier(cfg: dict) -> None:
    from training.train_classifier import train
    train(str(ROOT / "config.yaml"))


def _handle_prepare_dataset(cfg: dict) -> None:
    from training.prepare_dataset import main as prep_main
    prep_main()


def _handle_test(cfg: dict) -> None:
    from testing.test_runner import VideoTester
    from testing.evaluate import BenchmarkRunner
    from testing.report_generator import ReportGenerator

    print("Running video test suite...")
    tester = VideoTester()

    # Try to instantiate detector/classifier for testing
    app = FlightTrackerApp(cfg)
    app._setup_detector()
    app._setup_classifier()

    results = tester.run_all_tests(
        video_dir=str(ROOT / "testing" / "videos"),
        detector=app.detector,
        classifier=app.classifier,
    )

    out_path = ROOT / "testing" / "results" / f"test_results_{int(time.time())}.json"
    tester.save_results(results, str(out_path))
    print(f"Test results saved to {out_path}")


def _handle_generate_report(cfg: dict) -> None:
    from testing.report_generator import ReportGenerator
    from testing.evaluate import DetectionMetrics, ClassificationMetrics

    print("Generating report...")
    rg = ReportGenerator()
    det_metrics = DetectionMetrics(map50=0.0, map50_95=0.0, precision=0.0, recall=0.0, per_class={})
    cls_metrics = ClassificationMetrics(
        type_accuracy=0.0, airline_accuracy=0.0, variant_accuracy=0.0,
        top3_accuracy=0.0, type_f1_macro=0.0, per_class_accuracy={}
    )
    out_dir = str(ROOT / "reports")
    rg.generate_report(det_metrics, cls_metrics, [], out_dir)
    print(f"Report written to {out_dir}")


def main() -> None:
    args = parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cfg = load_config(args.config)

    # Sub-commands that don't need the live loop
    if args.prepare_dataset:
        _handle_prepare_dataset(cfg)
        return
    if args.train_detector:
        _handle_train_detector(cfg)
        return
    if args.train_classifier:
        _handle_train_classifier(cfg)
        return
    if args.test:
        _handle_test(cfg)
        return
    if args.generate_report:
        _handle_generate_report(cfg)
        return

    # Live detection loop
    app = FlightTrackerApp(cfg)
    app.setup()

    # Handle Ctrl+C gracefully
    def _sig_handler(sig, frame):
        logger.info("Signal %s received, stopping...", sig)
        app._running = False

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Parse source — numeric string to int for camera
    source = args.source
    if source.lstrip("-").isdigit():
        source = int(source)

    try:
        app.run(
            source=source,
            show_window=not args.no_window,
            web=args.web,
            skip_frames=args.skip_frames,
        )
    finally:
        app.print_summary()


if __name__ == "__main__":
    main()
