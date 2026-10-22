"""
Flask + SocketIO real-time web dashboard for FlightTracker CV.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    frame_queue: queue.Queue,
    detection_queue: queue.Queue,
    flight_tracker: Any,
    db_logger: Any,
    config: Optional[dict] = None,
) -> tuple:
    """
    Create and return the Flask app + SocketIO instance.

    frame_queue:     main loop pushes (annotated_frame, flight_data) tuples here
    detection_queue: main loop pushes new Detection events here
    flight_tracker:  instance of FlightRadar24Tracker / OpenSkyTracker
    db_logger:       FlightLogger instance
    """
    config = config or {}
    template_folder = str(Path(__file__).parent / "templates")
    static_folder = str(Path(__file__).parent / "static")

    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    app.config["SECRET_KEY"] = "flighttracker-cv-secret"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

    frame_quality = config.get("frame_quality", 70)
    _state = {"active_clients": 0}

    # ------------------------------------------------------------------
    # HTTP routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/detections")
    def api_detections():
        n = int(request.args.get("n", 20))
        rows = db_logger.get_recent(n) if db_logger else []
        return jsonify(rows)

    @app.route("/api/stats")
    def api_stats():
        stats = db_logger.get_stats() if db_logger else {}
        return jsonify(stats)

    @app.route("/api/flights")
    def api_flights():
        flights = flight_tracker.get_nearby_flights() if flight_tracker else []
        result = []
        for f in flights:
            result.append({
                "flight_id": f.flight_id,
                "callsign": f.callsign,
                "aircraft_type": f.aircraft_type,
                "airline_name": f.airline_name,
                "origin_iata": f.origin_iata,
                "dest_iata": f.dest_iata,
                "altitude_ft": f.altitude_ft,
                "speed_kts": f.speed_kts,
                "heading_deg": f.heading_deg,
                "lat": f.lat,
                "lon": f.lon,
            })
        return jsonify(result)

    @app.route("/api/config", methods=["POST"])
    def api_config():
        data = request.get_json(silent=True) or {}
        # Dynamically update config keys that are safe to change
        allowed_keys = {"sound_enabled", "tts_enabled", "screenshot_enabled",
                        "confidence_threshold", "show_trails"}
        updated = {}
        for k in allowed_keys:
            if k in data:
                config[k] = data[k]
                updated[k] = data[k]
        return jsonify({"status": "ok", "updated": updated})

    # ------------------------------------------------------------------
    # SocketIO events
    # ------------------------------------------------------------------

    @socketio.on("connect")
    def on_connect():
        _state["active_clients"] += 1
        logger.debug("Dashboard client connected. Total: %d", _state["active_clients"])

    @socketio.on("disconnect")
    def on_disconnect():
        _state["active_clients"] = max(0, _state["active_clients"] - 1)
        logger.debug("Dashboard client disconnected. Total: %d", _state["active_clients"])

    # ------------------------------------------------------------------
    # Background emitter task
    # ------------------------------------------------------------------

    def background_task():
        """
        Continuously pulls from frame_queue and detection_queue,
        encoding frames to JPEG and emitting to all connected clients.
        """
        last_frame_time = 0.0
        frame_interval = config.get("frame_emit_interval_ms", 100) / 1000.0

        while True:
            now = time.time()

            # Emit frame if interval elapsed
            if now - last_frame_time >= frame_interval:
                try:
                    annotated_frame = frame_queue.get_nowait()
                    _, buf = cv2.imencode(
                        ".jpg", annotated_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, frame_quality]
                    )
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    socketio.emit("frame", {"data": b64})
                    last_frame_time = now
                except queue.Empty:
                    pass
                except Exception as exc:
                    logger.debug("Frame emit error: %s", exc)

            # Emit detection events
            try:
                while True:
                    event = detection_queue.get_nowait()
                    event_type = event.pop("_type", "detection")
                    if event_type == "flight_matched":
                        socketio.emit("flight_matched", event)
                    else:
                        socketio.emit("detection", event)
            except queue.Empty:
                pass
            except Exception as exc:
                logger.debug("Detection emit error: %s", exc)

            socketio.sleep(0.01)

    socketio.start_background_task(background_task)

    return app, socketio


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_dashboard(
    frame_queue: queue.Queue,
    detection_queue: queue.Queue,
    flight_tracker: Any,
    db_logger: Any,
    host: str = "0.0.0.0",
    port: int = 5000,
    config: Optional[dict] = None,
) -> None:
    """
    Create and start the Flask/SocketIO dashboard server.
    This blocks until the server is stopped.
    """
    app, socketio = create_app(
        frame_queue, detection_queue, flight_tracker, db_logger, config
    )
    logger.info("Starting dashboard on http://%s:%d", host, port)
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
