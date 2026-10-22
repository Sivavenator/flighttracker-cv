"""
HUD overlay renderer for FlightTracker CV.

All drawing uses semi-transparent overlays via cv2.addWeighted so the
video feed remains visible beneath the HUD elements.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

@dataclass
class ColorScheme:
    """Day / night theme colours (BGR)."""
    detection_box: Tuple[int, int, int] = (0, 255, 0)
    detection_uncertain: Tuple[int, int, int] = (0, 165, 255)
    text_primary: Tuple[int, int, int] = (255, 255, 255)
    text_secondary: Tuple[int, int, int] = (200, 200, 200)
    hud_bg: Tuple[int, int, int] = (0, 0, 0)
    trail: Tuple[int, int, int] = (0, 200, 255)
    compass: Tuple[int, int, int] = (0, 200, 255)
    matched: Tuple[int, int, int] = (255, 50, 50)
    alert: Tuple[int, int, int] = (0, 0, 255)
    bar_bg: Tuple[int, int, int] = (50, 50, 50)


DAY_SCHEME = ColorScheme()
NIGHT_SCHEME = ColorScheme(
    detection_box=(0, 255, 150),
    detection_uncertain=(0, 200, 100),
    text_primary=(180, 255, 180),
    text_secondary=(120, 200, 120),
    hud_bg=(0, 10, 0),
    trail=(0, 180, 100),
    compass=(0, 200, 100),
    matched=(100, 255, 100),
    alert=(0, 100, 255),
)


# ---------------------------------------------------------------------------
# Main HUD renderer
# ---------------------------------------------------------------------------

class HUDRenderer:
    """
    Renders bounding boxes, flight info panels, trails, and the overall HUD
    onto a video frame.  All modifications are made on a copy of the frame.
    """

    TOP_BAR_H = 40
    BOTTOM_BAR_H = 40
    INFO_PANEL_W = 320
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_LG = 0.65
    FONT_SM = 0.48
    OPACITY = 0.75

    def __init__(self, scheme: Optional[ColorScheme] = None) -> None:
        self.scheme = scheme or DAY_SCHEME
        self._flash_timers: Dict[int, float] = {}  # track_id -> flash_end_time

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        label: str,
        confidence: float,
        colour: Optional[Tuple[int, int, int]] = None,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw a rounded-corner bounding box with label pill and confidence bar.
        """
        colour = colour or self.scheme.detection_box
        x1, y1, x2, y2 = [int(v) for v in bbox]
        overlay = frame.copy()

        # Rounded corners via drawing 4 corner arcs + edge rectangles
        r = 8
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), colour, 2)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), colour, 2)
        for cx, cy, start, end in [
            (x1 + r, y1 + r, 180, 270),
            (x2 - r, y1 + r, 270, 360),
            (x1 + r, y2 - r, 90,  180),
            (x2 - r, y2 - r, 0,    90),
        ]:
            cv2.ellipse(overlay, (cx, cy), (r, r), 0, start, end, colour, 2)

        # Blend the box overlay
        cv2.addWeighted(overlay, self.OPACITY, frame, 1 - self.OPACITY, 0, frame)

        # Label pill
        track_str = f"#{track_id} " if track_id is not None else ""
        full_label = f"{track_str}{label} {confidence:.0%}"
        (tw, th), bl = cv2.getTextSize(full_label, self.FONT, self.FONT_SM, 1)
        pill_x1, pill_y1 = x1, max(0, y1 - th - bl - 8)
        pill_x2, pill_y2 = x1 + tw + 10, y1

        pill_overlay = frame.copy()
        cv2.rectangle(pill_overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), colour, -1)
        cv2.addWeighted(pill_overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, full_label, (pill_x1 + 5, y1 - bl - 2),
                    self.FONT, self.FONT_SM, (0, 0, 0), 1, cv2.LINE_AA)

        # Confidence bar (bottom of box)
        bar_h = 4
        bar_w = x2 - x1
        conf_w = int(bar_w * confidence)
        cv2.rectangle(frame, (x1, y2 - bar_h), (x2, y2), self.scheme.bar_bg, -1)
        cv2.rectangle(frame, (x1, y2 - bar_h), (x1 + conf_w, y2), colour, -1)

        return frame

    # ------------------------------------------------------------------
    # Flight info panel
    # ------------------------------------------------------------------

    def draw_flight_info(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        flight_data: Any,
        classification: Any = None,
        distance_km: float = -1.0,
    ) -> np.ndarray:
        """
        Draw a flight information panel to the right of the bounding box.
        flight_data: Flight dataclass from flight_tracker.py
        classification: ClassificationResult from classifier.py
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Position panel to the right (or left if near edge)
        px = min(x2 + 8, w - self.INFO_PANEL_W - 4)
        py = max(y1, self.TOP_BAR_H + 4)
        pw = self.INFO_PANEL_W
        panel_lines: List[str] = []

        if flight_data:
            panel_lines.append(f"FLT  {flight_data.callsign or 'N/A'}")
            panel_lines.append(f"OPR  {flight_data.airline_name or flight_data.airline_iata or 'N/A'}")
            panel_lines.append(
                f"RTE  {flight_data.origin_iata or '???'} -> {flight_data.dest_iata or '???'}"
            )
            panel_lines.append(f"ALT  {int(flight_data.altitude_ft):,} ft")
            panel_lines.append(f"SPD  {int(flight_data.speed_kts)} kts")
            panel_lines.append(f"HDG  {int(flight_data.heading_deg):03d}°")

        if classification:
            panel_lines.append(f"TYPE {classification.aircraft_type}")
            panel_lines.append(f"VAR  {classification.variant}")
            panel_lines.append(f"CONF {classification.confidence:.0%}")

        if distance_km > 0:
            panel_lines.append(f"DIST ~{distance_km:.1f} km")

        if not panel_lines:
            return frame

        line_h = 22
        ph = line_h * len(panel_lines) + 16
        panel = frame.copy()
        cv2.rectangle(panel, (px, py), (px + pw, py + ph), self.scheme.hud_bg, -1)
        cv2.addWeighted(panel, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph),
                      self.scheme.detection_box, 1)

        for i, line in enumerate(panel_lines):
            ty = py + 14 + i * line_h
            cv2.putText(frame, line, (px + 6, ty),
                        self.FONT, self.FONT_SM, self.scheme.text_primary, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Trail
    # ------------------------------------------------------------------

    def draw_trail(
        self,
        frame: np.ndarray,
        track_history: List[Tuple[float, float]],
        colour: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw a fading dotted trail for a tracked object."""
        colour = colour or self.scheme.trail
        pts = list(track_history)
        if len(pts) < 2:
            return frame
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            faded = tuple(int(c * alpha) for c in colour)
            pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            pt2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(frame, pt1, pt2, faded, 1, cv2.LINE_AA)
            if i % 3 == 0:
                cv2.circle(frame, pt2, 2, faded, -1)
        return frame

    # ------------------------------------------------------------------
    # Full HUD
    # ------------------------------------------------------------------

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        detection_count: int = 0,
        location_name: str = "Singapore",
        fr24_connected: bool = False,
        bearing: float = 0.0,
    ) -> np.ndarray:
        """Draw top/bottom bars, compass, and signal indicator."""
        h, w = frame.shape[:2]

        # ---------- Top bar ----------
        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, self.TOP_BAR_H), self.scheme.hud_bg, -1)
        cv2.addWeighted(top_overlay, self.OPACITY, frame, 1 - self.OPACITY, 0, frame)

        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (10, 26),
                    self.FONT, self.FONT_LG, self.scheme.text_primary, 1, cv2.LINE_AA)
        cv2.putText(frame, location_name, (w // 2 - 60, 26),
                    self.FONT, self.FONT_SM, self.scheme.text_secondary, 1, cv2.LINE_AA)
        ac_str = f"AC: {detection_count}"
        (tw, _), _ = cv2.getTextSize(ac_str, self.FONT, self.FONT_LG, 1)
        cv2.putText(frame, ac_str, (w - tw - 10, 26),
                    self.FONT, self.FONT_LG, self.scheme.text_primary, 1, cv2.LINE_AA)

        # ---------- Bottom bar ----------
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (0, h - self.BOTTOM_BAR_H), (w, h), self.scheme.hud_bg, -1)
        cv2.addWeighted(bot_overlay, self.OPACITY, frame, 1 - self.OPACITY, 0, frame)

        fps_str = f"FPS {fps:.1f}"
        cv2.putText(frame, fps_str, (10, h - 12),
                    self.FONT, self.FONT_SM, self.scheme.text_primary, 1, cv2.LINE_AA)

        # FR24 signal indicator
        fr_colour = self.scheme.detection_box if fr24_connected else self.scheme.alert
        fr_str = "FR24 •" if fr24_connected else "FR24 ■"
        (ftw, _), _ = cv2.getTextSize(fr_str, self.FONT, self.FONT_SM, 1)
        cv2.putText(frame, fr_str, (w - ftw - 10, h - 12),
                    self.FONT, self.FONT_SM, fr_colour, 1, cv2.LINE_AA)

        # ---------- Compass ----------
        self.draw_compass(frame, bearing, position=(60, h // 2))

        return frame

    # ------------------------------------------------------------------
    # Aircraft info panel
    # ------------------------------------------------------------------

    def draw_aircraft_panel(
        self,
        frame: np.ndarray,
        aircraft_type: str = "Unknown",
        airline: str = "Unknown",
        origin: str = "Unknown",
        destination: str = "Unknown",
        accuracy: float = 0.0,
    ) -> np.ndarray:
        """Bottom-left info panel showing current aircraft details."""
        import cv2
        h, w = frame.shape[:2]
        panel_w, panel_h = 290, 168
        margin = 15
        x, y = margin, h - panel_h - margin

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Cyan border
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 229, 255), 1)

        # Header bar
        cv2.rectangle(frame, (x, y), (x + panel_w, y + 26), (0, 229, 255), -1)
        cv2.putText(frame, "  AIRCRAFT INFO", (x + 6, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # Accuracy bar
        bar_x = x + panel_w - 60
        bar_y = y + 9
        bar_w = 50
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (40, 40, 40), -1)
        fill = int(bar_w * min(accuracy / 100.0, 1.0))
        bar_color = (0, 200, 80) if accuracy >= 70 else (0, 165, 255) if accuracy >= 40 else (0, 0, 200)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 10), bar_color, -1)

        # Fields
        fields = [
            ("Type",     aircraft_type),
            ("Airline",  airline),
            ("From",     origin),
            ("To",       destination),
            ("Accuracy", f"{accuracy:.0f}%"),
        ]
        for i, (label, value) in enumerate(fields):
            fy = y + 44 + i * 24
            cv2.putText(frame, f"{label}:", (x + 10, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)
            col = (0, 255, 140) if value not in ("Unknown", "N/A", "") else (100, 100, 100)
            cv2.putText(frame, value or "Unknown", (x + 95, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

        return frame


    # ------------------------------------------------------------------
    # Compass rose
    # ------------------------------------------------------------------

    def draw_compass(
        self,
        frame: np.ndarray,
        bearing_deg: float,
        position: Tuple[int, int] = (60, 60),
    ) -> np.ndarray:
        """Draw a mini compass rose with a bearing needle."""
        cx, cy = position
        r = 35
        colour = self.scheme.compass

        compass_overlay = frame.copy()
        cv2.circle(compass_overlay, (cx, cy), r, self.scheme.hud_bg, -1)
        cv2.addWeighted(compass_overlay, 0.6, frame, 0.4, 0, frame)
        cv2.circle(frame, (cx, cy), r, colour, 1)

        # Cardinal labels
        for label, deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            rad = math.radians(deg - 90)
            lx = int(cx + (r - 8) * math.cos(rad))
            ly = int(cy + (r - 8) * math.sin(rad))
            cv2.putText(frame, label, (lx - 5, ly + 5),
                        self.FONT, 0.35, colour, 1, cv2.LINE_AA)

        # Bearing needle
        rad = math.radians(bearing_deg - 90)
        tip = (int(cx + (r - 6) * math.cos(rad)), int(cy + (r - 6) * math.sin(rad)))
        tail = (int(cx - 8 * math.cos(rad)), int(cy - 8 * math.sin(rad)))
        cv2.arrowedLine(frame, tail, tip, self.scheme.matched, 2,
                        cv2.LINE_AA, tipLength=0.3)

        # Centre dot
        cv2.circle(frame, (cx, cy), 3, colour, -1)
        return frame

    # ------------------------------------------------------------------
    # Detection flash
    # ------------------------------------------------------------------

    def animate_detection_flash(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw a brief green flash rectangle when a new detection appears.
        The flash fades out over 500 ms.
        """
        now = time.time()
        key = track_id if track_id is not None else -1
        if key not in self._flash_timers:
            self._flash_timers[key] = now + 0.5  # flash for 0.5 s

        end_time = self._flash_timers.get(key, 0)
        if now > end_time:
            return frame

        alpha = (end_time - now) / 0.5  # fade from 1 -> 0
        x1, y1, x2, y2 = [int(v) for v in bbox]
        flash = frame.copy()
        cv2.rectangle(flash, (x1, y1), (x2, y2), (0, 255, 100), -1)
        cv2.addWeighted(flash, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)
        return frame

    # ------------------------------------------------------------------
    # Night mode toggle
    # ------------------------------------------------------------------

    def set_night_mode(self, enabled: bool) -> None:
        self.scheme = NIGHT_SCHEME if enabled else DAY_SCHEME
