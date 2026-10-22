"""
Screenshot capture utility with EXIF metadata and timelapse export.
"""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScreenshotManager:
    """
    Captures annotated screenshots of detected aircraft, embeds metadata,
    and enforces a per-flight rate limit.
    """

    def __init__(
        self,
        output_dir: str = "screenshots",
        max_per_flight: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_flight = max_per_flight
        # flight_id -> count of screenshots taken
        self._counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------

    def capture(
        self,
        frame: np.ndarray,
        flight_data: Optional[Any],
        classification_result: Optional[Any],
    ) -> str:
        """
        Save an annotated screenshot.

        Respects max_per_flight rate limit per flight_id.
        Overlays metadata text on the saved image.
        Attempts to embed EXIF data using piexif.

        Returns the file path or empty string if rate-limited / failed.
        """
        flight_id = getattr(flight_data, "flight_id", "unknown") or "unknown"

        if self._counts[flight_id] >= self.max_per_flight:
            logger.debug("Screenshot rate limit reached for flight %s", flight_id)
            return ""

        now = datetime.now()
        callsign = (getattr(flight_data, "callsign", "") or "UNKN").strip().replace("/", "")
        aircraft_type = (getattr(classification_result, "aircraft_type", "") or "UNK").replace(" ", "")
        filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{callsign}_{aircraft_type}.jpg"
        filepath = self.output_dir / filename

        # Overlay metadata text onto a copy of the frame
        annotated = self._overlay_metadata(frame.copy(), flight_data, classification_result, now)

        # Save with JPEG quality 95
        ok = cv2.imwrite(str(filepath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            logger.warning("cv2.imwrite failed for %s", filepath)
            return ""

        # Attempt EXIF embedding
        self._embed_exif(filepath, flight_data, classification_result, now)

        self._counts[flight_id] += 1
        logger.info("Screenshot saved: %s (%d/%d for flight %s)",
                    filepath, self._counts[flight_id], self.max_per_flight, flight_id)
        return str(filepath)

    # ------------------------------------------------------------------

    def _overlay_metadata(
        self,
        frame: np.ndarray,
        flight_data: Optional[Any],
        classification_result: Optional[Any],
        dt: datetime,
    ) -> np.ndarray:
        """Overlay metadata text in the top-left corner."""
        lines = [dt.strftime("%Y-%m-%d %H:%M:%S UTC")]
        if classification_result:
            lines.append(f"Type: {classification_result.aircraft_type} ({classification_result.variant})")
            lines.append(f"Airline: {classification_result.airline}")
        if flight_data:
            lines.append(f"Callsign: {getattr(flight_data, 'callsign', 'N/A')}")
            lines.append(
                f"Route: {getattr(flight_data, 'origin_iata', '?')} "
                f"-> {getattr(flight_data, 'dest_iata', '?')}"
            )
            lines.append(f"Alt: {int(getattr(flight_data, 'altitude_ft', 0)):,} ft  "
                         f"Spd: {int(getattr(flight_data, 'speed_kts', 0))} kts")

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        y = 24
        for line in lines:
            (tw, th), bl = cv2.getTextSize(line, font, scale, thickness)
            cv2.rectangle(frame, (8, y - th - 2), (8 + tw + 4, y + bl + 2), (0, 0, 0), -1)
            cv2.putText(frame, line, (10, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += th + bl + 6

        return frame

    def _embed_exif(
        self,
        filepath: Path,
        flight_data: Optional[Any],
        classification_result: Optional[Any],
        dt: datetime,
    ) -> None:
        """Embed EXIF metadata using piexif (best-effort)."""
        try:
            import piexif

            exif_dict = {
                "0th": {},
                "Exif": {},
                "GPS": {},
                "1st": {},
                "thumbnail": None,
            }

            # DateTime
            dt_str = dt.strftime("%Y:%m:%d %H:%M:%S").encode("utf-8")
            exif_dict["0th"][piexif.ImageIFD.DateTime] = dt_str

            # Artist field = aircraft type + airline
            if classification_result:
                info = (
                    f"{classification_result.aircraft_type} | "
                    f"{classification_result.airline}"
                ).encode("utf-8")
                exif_dict["0th"][piexif.ImageIFD.Artist] = info

            # ImageDescription = callsign + route
            if flight_data:
                desc = (
                    f"{getattr(flight_data, 'callsign', '')} "
                    f"{getattr(flight_data, 'origin_iata', '')} "
                    f"-> {getattr(flight_data, 'dest_iata', '')}"
                ).encode("utf-8")
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = desc

            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(filepath))
        except ImportError:
            pass  # piexif not installed; skip silently
        except Exception as exc:
            logger.debug("EXIF embedding failed: %s", exc)

    # ------------------------------------------------------------------

    def create_timelapse(self, flight_id: str, fps: int = 5) -> str:
        """
        Combine screenshots for a given flight_id into an MP4 timelapse.
        Returns the output video path, or empty string if fewer than 2 frames.
        """
        # Gather matching screenshots
        frames = sorted(self.output_dir.glob(f"*_{flight_id}_*.jpg"))
        if len(frames) < 2:
            logger.info("Not enough screenshots for timelapse of flight %s", flight_id)
            return ""

        # Read first frame to get dimensions
        sample = cv2.imread(str(frames[0]))
        if sample is None:
            return ""
        h, w = sample.shape[:2]

        out_path = self.output_dir / f"timelapse_{flight_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                writer.write(img)

        writer.release()
        logger.info("Timelapse saved: %s (%d frames)", out_path, len(frames))
        return str(out_path)

    def reset_counts(self) -> None:
        """Clear per-flight screenshot counts (e.g. at start of new session)."""
        self._counts.clear()
