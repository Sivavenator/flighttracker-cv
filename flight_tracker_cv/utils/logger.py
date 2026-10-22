"""
SQLite-backed flight detection logger using SQLAlchemy ORM.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLAlchemy ORM model
# ---------------------------------------------------------------------------

def _get_engine_and_base(db_path: str):
    """Create SQLAlchemy engine + Base for the given SQLite path."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False})
    Base = declarative_base()
    return engine, Base


def _build_model(Base):
    from sqlalchemy import Column, Integer, String, Float, DateTime

    class FlightDetection(Base):
        __tablename__ = "flight_detections"

        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        aircraft_type = Column(String(64), default="")
        variant = Column(String(64), default="")
        airline = Column(String(128), default="")
        callsign = Column(String(32), default="")
        origin_iata = Column(String(8), default="")
        origin_name = Column(String(128), default="")
        dest_iata = Column(String(8), default="")
        dest_name = Column(String(128), default="")
        altitude_ft = Column(Float, default=0.0)
        speed_kts = Column(Float, default=0.0)
        heading_deg = Column(Float, default=0.0)
        confidence_type = Column(Float, default=0.0)
        confidence_airline = Column(Float, default=0.0)
        distance_km = Column(Float, default=-1.0)
        bearing_deg = Column(Float, default=0.0)
        screenshot_path = Column(String(512), default="")
        flight_id = Column(String(64), default="")
        registration = Column(String(32), default="")

        def to_dict(self) -> Dict[str, Any]:
            return {
                "id": self.id,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "aircraft_type": self.aircraft_type,
                "variant": self.variant,
                "airline": self.airline,
                "callsign": self.callsign,
                "origin_iata": self.origin_iata,
                "origin_name": self.origin_name,
                "dest_iata": self.dest_iata,
                "dest_name": self.dest_name,
                "altitude_ft": self.altitude_ft,
                "speed_kts": self.speed_kts,
                "heading_deg": self.heading_deg,
                "confidence_type": self.confidence_type,
                "confidence_airline": self.confidence_airline,
                "distance_km": self.distance_km,
                "bearing_deg": self.bearing_deg,
                "screenshot_path": self.screenshot_path,
                "flight_id": self.flight_id,
                "registration": self.registration,
            }

    return FlightDetection


# ---------------------------------------------------------------------------
# FlightLogger
# ---------------------------------------------------------------------------

class FlightLogger:
    """
    Persists aircraft detection events to a SQLite database.
    Thread-safe for use from the main CV loop and background threads.
    """

    def __init__(self, db_path: str = "logs/flight_detections.db") -> None:
        self.db_path = db_path
        self._engine, Base = _get_engine_and_base(db_path)
        self.FlightDetection = _build_model(Base)
        Base.metadata.create_all(self._engine)

        from sqlalchemy.orm import sessionmaker
        self._Session = sessionmaker(bind=self._engine)
        logger.info("FlightLogger initialised at %s", db_path)

    # ------------------------------------------------------------------

    def log_detection(
        self,
        flight_data: Optional[Any],
        classification_result: Optional[Any],
        distance_km: float = -1.0,
        bearing_deg: float = 0.0,
        screenshot_path: str = "",
    ) -> int:
        """
        Persist one detection event. Returns the new record's ID.
        flight_data: Flight dataclass (may be None if FR24 not matched)
        classification_result: ClassificationResult (may be None)
        """
        session = self._Session()
        try:
            rec = self.FlightDetection(
                timestamp=datetime.utcnow(),
                aircraft_type=getattr(classification_result, "aircraft_type", "") or "",
                variant=getattr(classification_result, "variant", "") or "",
                airline=getattr(classification_result, "airline", "") or "",
                callsign=getattr(flight_data, "callsign", "") or "",
                origin_iata=getattr(flight_data, "origin_iata", "") or "",
                origin_name=getattr(flight_data, "origin_name", "") or "",
                dest_iata=getattr(flight_data, "dest_iata", "") or "",
                dest_name=getattr(flight_data, "dest_name", "") or "",
                altitude_ft=float(getattr(flight_data, "altitude_ft", 0) or 0),
                speed_kts=float(getattr(flight_data, "speed_kts", 0) or 0),
                heading_deg=float(getattr(flight_data, "heading_deg", 0) or 0),
                confidence_type=float(getattr(classification_result, "confidence", 0) or 0),
                confidence_airline=0.0,
                distance_km=distance_km,
                bearing_deg=bearing_deg,
                screenshot_path=screenshot_path,
                flight_id=getattr(flight_data, "flight_id", "") or "",
                registration=getattr(flight_data, "registration", "") or "",
            )
            session.add(rec)
            session.commit()
            new_id = rec.id
            logger.debug("Logged detection id=%d callsign=%s", new_id, rec.callsign)
            return new_id
        except Exception as exc:
            session.rollback()
            logger.error("Log error: %s", exc)
            return -1
        finally:
            session.close()

    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        session = self._Session()
        try:
            from sqlalchemy import func, distinct
            today = date.today()

            total = session.query(func.count(self.FlightDetection.id)).scalar() or 0
            unique_flights = session.query(
                func.count(distinct(self.FlightDetection.flight_id))
            ).scalar() or 0
            airlines_seen = session.query(
                func.count(distinct(self.FlightDetection.airline))
            ).scalar() or 0
            types_seen = session.query(
                func.count(distinct(self.FlightDetection.aircraft_type))
            ).scalar() or 0
            today_count = session.query(func.count(self.FlightDetection.id)).filter(
                func.date(self.FlightDetection.timestamp) == today
            ).scalar() or 0

            return {
                "total_detections": total,
                "unique_flights": unique_flights,
                "airlines_seen": airlines_seen,
                "types_seen": types_seen,
                "today_count": today_count,
            }
        finally:
            session.close()

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the n most-recent detection records as dicts."""
        session = self._Session()
        try:
            rows = (
                session.query(self.FlightDetection)
                .order_by(self.FlightDetection.id.desc())
                .limit(n)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    # ------------------------------------------------------------------

    def export_csv(self, output_path: str) -> None:
        """Export all detections to CSV."""
        records = self.get_recent(n=100_000)
        if not records:
            logger.info("No records to export.")
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        logger.info("Exported %d records to %s", len(records), output_path)

    def export_json(self, output_path: str) -> None:
        """Export all detections to JSON."""
        records = self.get_recent(n=100_000)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
        logger.info("Exported %d records to %s", len(records), output_path)
