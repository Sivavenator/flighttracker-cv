"""
YOLOv8 aircraft detector with ByteTrack multi-object tracking.
"""

from __future__ import annotations

import math
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single aircraft detection result."""
    bbox_xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    frame_idx: int = 0


class AircraftDetector:
    """
    YOLOv8-based aircraft detector with ByteTrack multi-object tracking.

    Wraps the Ultralytics YOLO API to detect and track aircraft in video frames,
    compute per-detection distance/bearing estimates, and render HUD overlays.
    """

    # Colour palette for bounding boxes (per track_id % len)
    BOX_COLOURS = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 160, 0),
        (200, 0, 255),
        (255, 0, 120),
        (0, 255, 180),
        (255, 255, 0),
        (100, 255, 100),
    ]

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        max_detections: int = 20,
        use_tracker: bool = True,
        focal_length_px: Optional[float] = None,
        hfov_degrees: float = 60.0,
    ) -> None:
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.use_tracker = use_tracker
        self.focal_length_px: Optional[float] = focal_length_px
        self.hfov_degrees = hfov_degrees

        # Per-track bounding-box history for trail rendering
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.frame_idx: int = 0

        self.device = self._resolve_device(device)
        self.model = self._load_model(weights_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self, weights_path: Optional[str]):
        """Load YOLO model with graceful CPU fallback."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        if weights_path and Path(weights_path).exists():
            model_source = weights_path
            logger.info("Loading custom weights from %s", weights_path)
        else:
            model_source = "yolov8m.pt"
            logger.info(
                "Custom weights not found; loading pre-trained yolov8m.pt "
                "(will detect general objects, not aircraft-specific classes)"
            )

        try:
            model = YOLO(model_source)
            model.to(self.device)
            logger.info("YOLO model loaded on device: %s", self.device)
        except Exception as exc:
            logger.warning(
                "Failed to load model on %s (%s). Falling back to CPU.", self.device, exc
            )
            self.device = "cpu"
            model = YOLO(model_source)
            model.to("cpu")

        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection (+ optional tracking) on a single BGR frame.

        Returns a list of Detection namedtuples with track IDs assigned when
        tracking is enabled.
        """
        self.frame_idx += 1
        h, w = frame.shape[:2]

        if self.use_tracker:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            try:
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                track_id: Optional[int] = None
                if self.use_tracker and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                    # Record centre point for trail rendering
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    self.track_history[track_id].append((cx, cy))

                detections.append(
                    Detection(
                        bbox_xyxy=tuple(xyxy),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        track_id=track_id,
                        frame_idx=self.frame_idx,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error parsing detection %d: %s", i, exc)
                continue

        return detections

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_track_id: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of the frame.
        Returns the annotated frame (does NOT modify in-place).
        """
        out = frame.copy()
        for det in detections:
            colour = self._colour_for_track(det.track_id)
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

            # Main box
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            # Label background + text
            parts = [det.class_name]
            if show_confidence:
                parts.append(f"{det.confidence:.2f}")
            if show_track_id and det.track_id is not None:
                parts.insert(0, f"#{det.track_id}")
            label = " ".join(parts)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            lx, ly = x1, max(y1 - 4, th + 4)
            cv2.rectangle(out, (lx, ly - th - baseline), (lx + tw + 4, ly + 2), colour, -1)
            cv2.putText(
                out, label, (lx + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )

            # Confidence bar inside box (bottom 6px strip)
            bar_w = int((x2 - x1) * det.confidence)
            cv2.rectangle(out, (x1, y2 - 6), (x1 + bar_w, y2), colour, -1)

        return out

    def draw_trail(
        self,
        frame: np.ndarray,
        track_id: int,
        colour: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw dotted movement trail for a given track ID."""
        out = frame
        history = self.track_history.get(track_id)
        if not history or len(history) < 2:
            return out

        pts = list(history)
        if colour is None:
            colour = self._colour_for_track(track_id)

        for idx in range(1, len(pts)):
            alpha = idx / len(pts)
            faded = tuple(int(c * alpha) for c in colour)
            pt1 = (int(pts[idx - 1][0]), int(pts[idx - 1][1]))
            pt2 = (int(pts[idx][0]), int(pts[idx][1]))
            cv2.line(frame, pt1, pt2, faded, 1, cv2.LINE_AA)
            if idx % 3 == 0:
                cv2.circle(frame, pt2, 2, faded, -1)
        return out

    # ------------------------------------------------------------------
    # Distance / bearing estimation
    # ------------------------------------------------------------------

    def estimate_distance(
        self,
        bbox: Tuple[float, float, float, float],
        known_wingspan_m: float = 60.0,
        focal_length_px: Optional[float] = None,
        frame_width: Optional[int] = None,
    ) -> float:
        """
        Pinhole-camera distance estimate using apparent bounding-box width.

        distance_m = (known_wingspan_m * focal_length_px) / bbox_width_px

        Returns distance in kilometres.  Returns -1.0 if focal length is unknown.
        """
        fl = focal_length_px or self.focal_length_px
        if fl is None:
            if frame_width is not None:
                # Approximate: assume HFOV and compute focal length
                fl = (frame_width / 2) / math.tan(math.radians(self.hfov_degrees / 2))
                self.focal_length_px = fl
            else:
                return -1.0

        x1, y1, x2, y2 = bbox
        bbox_width_px = max(x2 - x1, 1)
        distance_m = (known_wingspan_m * fl) / bbox_width_px
        return round(distance_m / 1000.0, 2)

    def get_bearing_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        hfov_degrees: Optional[float] = None,
        camera_heading_deg: float = 0.0,
    ) -> float:
        """
        Estimate compass bearing to aircraft from its image-plane position.

        Assumes camera is level (no pitch/roll correction).
        hfov_degrees: horizontal field of view of the camera lens.
        camera_heading_deg: direction the camera is pointing (0 = North).

        Returns bearing in [0, 360).
        """
        hfov = hfov_degrees or self.hfov_degrees
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0

        # Angle offset from image centre
        offset_px = cx - frame_width / 2.0
        angle_per_px = hfov / frame_width
        offset_deg = offset_px * angle_per_px

        bearing = (camera_heading_deg + offset_deg) % 360.0
        return round(bearing, 1)

    # ------------------------------------------------------------------
    # Calibration helper
    # ------------------------------------------------------------------

    def auto_calibrate_focal_length(
        self,
        ref_distance_m: float,
        ref_bbox: Tuple[float, float, float, float],
        known_wingspan_m: float = 60.0,
    ) -> float:
        """
        Calibrate focal length given a reference aircraft at a known distance.

        Stores result in self.focal_length_px and returns it.
        """
        x1, _, x2, _ = ref_bbox
        bbox_width_px = max(x2 - x1, 1)
        fl = (bbox_width_px * ref_distance_m) / known_wingspan_m
        self.focal_length_px = fl
        logger.info("Auto-calibrated focal length: %.1f px", fl)
        return fl

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _colour_for_track(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        if track_id is None:
            return self.BOX_COLOURS[0]
        return self.BOX_COLOURS[track_id % len(self.BOX_COLOURS)]

    def reset_tracker(self) -> None:
        """Clear all track history (call when changing video source)."""
        self.track_history.clear()
        self.frame_idx = 0

    def get_active_track_ids(self) -> List[int]:
        """Return list of track IDs seen recently."""
        return list(self.track_history.keys())
