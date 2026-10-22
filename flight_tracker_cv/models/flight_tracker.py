"""
FlightRadar24 and OpenSky Network integration for live flight data.

Provides a background-thread polling model so the main CV loop is never
blocked waiting for network I/O.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flight dataclass
# ---------------------------------------------------------------------------

@dataclass
class Flight:
    flight_id: str
    callsign: str
    aircraft_type: str        # ICAO type code e.g. "A350"
    airline_iata: str
    airline_name: str
    origin_iata: str
    origin_name: str
    origin_city: str
    dest_iata: str
    dest_name: str
    dest_city: str
    altitude_ft: float
    speed_kts: float
    heading_deg: float
    lat: float
    lon: float
    registration: str
    squawk: str
    last_updated: float = field(default_factory=time.time)


@dataclass
class MatchResult:
    """Returned by match_to_detection — wraps the matched flight with metadata."""
    flight: Flight
    score: float               # 0.0 – 1.0; higher = more confident
    bearing_error_deg: float   # how far off the bearing was
    estimated_distance_km: float


def _normalise_type(raw: str) -> str:
    """
    Normalise an aircraft type string for fuzzy comparison.
    'Boeing 737-800' -> 'B738', 'A350-900' -> 'A359', 'B777' -> 'B777'
    Handles both YOLO label format and FR24 ICAO codes.
    """
    if not raw:
        return ""
    s = raw.upper().replace(" ", "").replace("-", "").replace("_", "")
    # Map common verbose names to ICAO-style codes
    _MAP = {
        "BOEING737": "B737", "BOEING747": "B747", "BOEING757": "B757",
        "BOEING767": "B767", "BOEING777": "B777", "BOEING787": "B787",
        "AIRBUS320": "A320", "AIRBUS321": "A321", "AIRBUS319": "A319",
        "AIRBUS330": "A330", "AIRBUS340": "A340", "AIRBUS350": "A350",
        "AIRBUS380": "A380", "AIRBUS220": "A220",
        "7378": "B737", "7888": "B788", "7898": "B789",
        "A359": "A350", "A35K": "A350",
        "B738": "B737", "B739": "B737", "B77W": "B777", "B77L": "B777",
        "B788": "B787", "B789": "B787", "B78X": "B787",
        "A21N": "A321", "A20N": "A320", "A19N": "A319",
        "A339": "A330", "A338": "A330",
    }
    for key, val in _MAP.items():
        if key in s:
            return val
    # Return first 4 chars as a best-effort family code
    return s[:4]


# ---------------------------------------------------------------------------
# Airline database
# ---------------------------------------------------------------------------

class AirlineDatabase:
    """IATA code → airline metadata, loaded from airline_codes.json."""

    def __init__(self, json_path: Optional[str] = None) -> None:
        self._db: Dict[str, Dict] = {}
        if json_path and Path(json_path).exists():
            self._load(json_path)
        else:
            self._load_builtin()

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            self._db = json.load(fh)
        logger.info("Loaded %d airlines from %s", len(self._db), path)

    def _load_builtin(self) -> None:
        self._db = {
            "SQ": {"name": "Singapore Airlines", "country": "Singapore", "icao": "SIA"},
            "EK": {"name": "Emirates", "country": "UAE", "icao": "UAE"},
            "QF": {"name": "Qantas", "country": "Australia", "icao": "QFA"},
            "BA": {"name": "British Airways", "country": "UK", "icao": "BAW"},
            "LH": {"name": "Lufthansa", "country": "Germany", "icao": "DLH"},
            "AF": {"name": "Air France", "country": "France", "icao": "AFR"},
            "UA": {"name": "United Airlines", "country": "USA", "icao": "UAL"},
            "AA": {"name": "American Airlines", "country": "USA", "icao": "AAL"},
            "DL": {"name": "Delta Air Lines", "country": "USA", "icao": "DAL"},
            "CX": {"name": "Cathay Pacific", "country": "Hong Kong", "icao": "CPA"},
        }

    def get_name(self, iata: str) -> str:
        return self._db.get(iata, {}).get("name", iata)

    def get_info(self, iata: str) -> Dict:
        return self._db.get(iata, {"name": iata, "country": "Unknown"})


# ---------------------------------------------------------------------------
# FlightRadar24 implementation
# ---------------------------------------------------------------------------

class FlightRadar24Tracker:
    """
    Polls FlightRadarAPI for aircraft near a given lat/lon.
    Runs in a background daemon thread.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        update_interval_s: float = 5.0,
        airline_db: Optional[AirlineDatabase] = None,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.radius_km = radius_km
        self.update_interval_s = update_interval_s
        self.airline_db = airline_db or AirlineDatabase()

        self._flights: List[Flight] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fr24 = None
        self._init_fr24()

    def _init_fr24(self) -> None:
        try:
            from FlightRadar24 import FlightRadar24API  # type: ignore
            self._fr24 = FlightRadar24API()
            logger.info("FlightRadarAPI initialised successfully.")
        except ImportError:
            logger.warning("FlightRadar24 package not installed. FR24 tracking disabled.")
        except Exception as exc:
            logger.warning("FR24 init error: %s", exc)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="FR24-poller")
        self._thread.start()
        logger.info("FR24 background polling started (interval=%ss).", self.update_interval_s)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("FR24 polling stopped.")

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._refresh()
            except Exception as exc:  # noqa: BLE001
                logger.error("FR24 poll error: %s", exc)
            time.sleep(self.update_interval_s)

    def _refresh(self) -> None:
        if self._fr24 is None:
            return

        # Compute bounding box from centre + radius
        delta_lat = self.radius_km / 111.0
        delta_lon = self.radius_km / (111.0 * math.cos(math.radians(self.lat)))

        bounds = self._fr24.get_bounds(
            self.lat + delta_lat,
            self.lat - delta_lat,
            self.lon - delta_lon,
            self.lon + delta_lon,
        )

        raw_flights = self._fr24.get_flights(bounds=bounds)
        parsed: List[Flight] = []
        for rf in raw_flights:
            try:
                airline_iata = getattr(rf, "airline_iata", "") or ""
                airline_name = self.airline_db.get_name(airline_iata) if airline_iata else ""
                f = Flight(
                    flight_id=str(getattr(rf, "id", "")),
                    callsign=str(getattr(rf, "callsign", "") or "").strip(),
                    aircraft_type=str(getattr(rf, "aircraft_code", "") or ""),
                    airline_iata=airline_iata,
                    airline_name=airline_name,
                    origin_iata=str(getattr(rf, "origin_airport_iata", "") or ""),
                    origin_name=str(getattr(rf, "origin_airport_name", "") or ""),
                    origin_city=str(getattr(rf, "origin_city", "") or ""),
                    dest_iata=str(getattr(rf, "destination_airport_iata", "") or ""),
                    dest_name=str(getattr(rf, "destination_airport_name", "") or ""),
                    dest_city=str(getattr(rf, "destination_city", "") or ""),
                    altitude_ft=float(getattr(rf, "altitude", 0) or 0),
                    speed_kts=float(getattr(rf, "ground_speed", 0) or 0),
                    heading_deg=float(getattr(rf, "heading", 0) or 0),
                    lat=float(getattr(rf, "latitude", 0) or 0),
                    lon=float(getattr(rf, "longitude", 0) or 0),
                    registration=str(getattr(rf, "registration", "") or ""),
                    squawk=str(getattr(rf, "squawk", "") or ""),
                )
                parsed.append(f)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to parse FR24 flight: %s", exc)

        with self._lock:
            self._flights = parsed
        logger.debug("FR24 refreshed: %d flights nearby.", len(parsed))

    def get_nearby_flights(self) -> List[Flight]:
        with self._lock:
            return list(self._flights)

    def get_flight_details(self, flight_id: str) -> Optional[Flight]:
        """Fetch enriched flight details for a specific flight_id."""
        with self._lock:
            for f in self._flights:
                if f.flight_id == flight_id:
                    return f
        if self._fr24 is None:
            return None
        try:
            details = self._fr24.get_flight_details(flight_id)
            if not details:
                return None
            # Enrich basic flight object with origin/dest names if available
            with self._lock:
                for f in self._flights:
                    if f.flight_id == flight_id:
                        trail = details.get("trail", [])
                        _ = trail  # trail is available but not used here directly
                        return f
        except Exception as exc:
            logger.warning("get_flight_details error: %s", exc)
        return None

    def match_to_detection(
        self,
        detection_bearing_deg: float,
        detection_distance_km: float,
        detected_type: str,
        bearing_tolerance_deg: float = 8.0,
        min_score: float = 0.55,
    ) -> Optional["MatchResult"]:
        """
        Bearing-first matching strategy.

        Because the camera is at a fixed, known location, every pixel column
        in the frame maps to a unique compass bearing.  We use that bearing as
        the PRIMARY signal and treat everything else as validation.

        Algorithm
        ---------
        1. Filter: keep only flights whose computed bearing from the camera
           lies within `bearing_tolerance_deg` of `detection_bearing_deg`.
           This alone removes ~95 % of candidates.
        2. Filter: discard aircraft below min altitude (ground traffic) or
           too slow (taxiing).
        3. Score the survivors on:
              bearing accuracy   60 %   (how well the bearing lines up)
              type match         25 %   (YOLO type vs FR24 ICAO type code)
              distance match     15 %   (bbox size estimate vs FR24 distance)
        4. Return the best scorer above `min_score`, or None.

        Returns a MatchResult namedtuple with the Flight plus match metadata.
        """
        flights = self.get_nearby_flights()
        if not flights:
            return None

        best_flight: Optional[Flight] = None
        best_score: float = -1.0
        best_bear_diff: float = 999.0
        best_dist_km: float = 0.0

        for flight in flights:
            # ── Plausibility filter ──────────────────────────────────────────
            if flight.altitude_ft < 500:        # skip ground traffic
                continue
            if flight.speed_kts < 80:           # skip taxiing / parked
                continue

            # ── Compute actual bearing from camera to this flight ────────────
            if not (self.lat and self.lon and flight.lat and flight.lon):
                continue
            flight_bearing = _bearing_between(self.lat, self.lon, flight.lat, flight.lon)

            bear_diff = abs(flight_bearing - detection_bearing_deg) % 360
            if bear_diff > 180:
                bear_diff = 360 - bear_diff

            # ── Primary gate: reject if outside bearing window ───────────────
            if bear_diff > bearing_tolerance_deg:
                continue

            # ── Score remaining candidates ───────────────────────────────────
            score = 0.0

            # 1. Bearing accuracy (60 %) — tighter match = higher score
            bear_score = max(0.0, 1.0 - bear_diff / bearing_tolerance_deg)
            score += 0.60 * bear_score

            # 2. Aircraft type match (25 %)
            #    Compare YOLO-detected type string against FR24 ICAO type code.
            #    E.g. "A350" matches "A359" / "A35K"; "B737" matches "B738".
            detected_norm = _normalise_type(detected_type)
            flight_norm   = _normalise_type(flight.aircraft_type)
            if detected_norm and flight_norm:
                if detected_norm == flight_norm:
                    type_score = 1.0
                elif detected_norm[:3] == flight_norm[:3]:   # family match
                    type_score = 0.7
                elif detected_norm[:2] == flight_norm[:2]:   # manufacturer
                    type_score = 0.4
                else:
                    type_score = 0.0
            else:
                type_score = 0.3   # unknown type — neutral bonus
            score += 0.25 * type_score

            # 3. Distance plausibility (15 %)
            actual_dist_km = _haversine(self.lat, self.lon, flight.lat, flight.lon)
            if detection_distance_km > 0:
                ratio = min(detection_distance_km, actual_dist_km) / max(
                    detection_distance_km, actual_dist_km, 0.1
                )
                dist_score = ratio          # 1.0 = perfect match
            else:
                dist_score = 0.5            # no estimate — neutral
            score += 0.15 * dist_score

            if score > best_score:
                best_score = score
                best_flight = flight
                best_bear_diff = bear_diff
                best_dist_km = actual_dist_km

        if best_score < min_score or best_flight is None:
            return None

        return MatchResult(
            flight=best_flight,
            score=best_score,
            bearing_error_deg=best_bear_diff,
            estimated_distance_km=best_dist_km,
        )

    def match_to_detection_bulk(
        self,
        detections: list,  # list of (bearing_deg, distance_km, type_str)
        bearing_tolerance_deg: float = 8.0,
    ) -> dict:
        """
        Match multiple detections in one pass — avoids redundant FR24 lookups.
        Returns dict mapping index -> MatchResult (or None).
        """
        flights = self.get_nearby_flights()
        if not flights:
            return {i: None for i in range(len(detections))}

        results = {}
        for i, (bearing_deg, distance_km, det_type) in enumerate(detections):
            results[i] = self.match_to_detection(
                bearing_deg, distance_km, det_type, bearing_tolerance_deg
            )
        return results


# ---------------------------------------------------------------------------
# OpenSky fallback
# ---------------------------------------------------------------------------

class OpenSkyTracker:
    """
    Polls the OpenSky Network REST API as a fallback to FR24.
    Same public interface as FlightRadar24Tracker.
    """

    _BASE_URL = "https://opensky-network.org/api/states/all"

    def __init__(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        update_interval_s: float = 10.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        airline_db: Optional[AirlineDatabase] = None,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.radius_km = radius_km
        self.update_interval_s = update_interval_s
        self.username = username
        self.password = password
        self.airline_db = airline_db or AirlineDatabase()

        self._flights: List[Flight] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="OpenSky-poller")
        self._thread.start()
        logger.info("OpenSky polling started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._refresh()
            except Exception as exc:
                logger.error("OpenSky poll error: %s", exc)
            time.sleep(self.update_interval_s)

    def _refresh(self) -> None:
        delta = self.radius_km / 111.0
        params = {
            "lamin": self.lat - delta,
            "lamax": self.lat + delta,
            "lomin": self.lon - delta,
            "lomax": self.lon + delta,
        }
        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)

        resp = requests.get(self._BASE_URL, params=params, auth=auth, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        states = data.get("states") or []

        parsed: List[Flight] = []
        for s in states:
            # OpenSky state vector fields (indices):
            # 0:icao24 1:callsign 2:origin_country 3:time_position 4:last_contact
            # 5:longitude 6:latitude 7:baro_altitude 8:on_ground 9:velocity
            # 10:true_track(heading) 11:vertical_rate 12:sensors 13:geo_altitude
            # 14:squawk 15:spi 16:position_source
            try:
                if s[8]:  # on_ground
                    continue
                lon_s = s[5] if s[5] else 0.0
                lat_s = s[6] if s[6] else 0.0
                alt_m = s[7] if s[7] else 0.0
                speed_ms = s[9] if s[9] else 0.0
                heading = s[10] if s[10] else 0.0
                callsign = (s[1] or "").strip()
                squawk = str(s[14] or "")

                f = Flight(
                    flight_id=str(s[0]),
                    callsign=callsign,
                    aircraft_type="",
                    airline_iata="",
                    airline_name="",
                    origin_iata="",
                    origin_name="",
                    origin_city="",
                    dest_iata="",
                    dest_name="",
                    dest_city="",
                    altitude_ft=alt_m * 3.281,
                    speed_kts=speed_ms * 1.944,
                    heading_deg=heading,
                    lat=lat_s,
                    lon=lon_s,
                    registration="",
                    squawk=squawk,
                )
                parsed.append(f)
            except (IndexError, TypeError):
                continue

        with self._lock:
            self._flights = parsed
        logger.debug("OpenSky refreshed: %d flights.", len(parsed))

    def get_nearby_flights(self) -> List[Flight]:
        with self._lock:
            return list(self._flights)

    def get_flight_details(self, flight_id: str) -> Optional[Flight]:
        with self._lock:
            for f in self._flights:
                if f.flight_id == flight_id:
                    return f
        return None

    def match_to_detection(
        self,
        detection_bearing_deg: float,
        detection_distance_km: float,
        detected_type: str,
    ) -> Optional[Flight]:
        flights = self.get_nearby_flights()
        if not flights:
            return None
        best: Optional[Flight] = None
        best_score = -1.0
        for flight in flights:
            hdg_diff = abs(flight.heading_deg - detection_bearing_deg) % 360
            if hdg_diff > 180:
                hdg_diff = 360 - hdg_diff
            hdg_score = max(0.0, 1.0 - hdg_diff / 180.0)
            if flight.lat and flight.lon:
                fb = _bearing_between(self.lat, self.lon, flight.lat, flight.lon)
                bear_diff = abs(fb - detection_bearing_deg) % 360
                if bear_diff > 180:
                    bear_diff = 360 - bear_diff
                bear_score = max(0.0, 1.0 - bear_diff / 90.0)
                dist = _haversine(self.lat, self.lon, flight.lat, flight.lon)
                if detection_distance_km > 0:
                    dist_score = max(0.0, 1.0 - abs(dist - detection_distance_km) / max(detection_distance_km, 1))
                else:
                    dist_score = 0.5
            else:
                bear_score = 0.5
                dist_score = 0.5
            score = 0.5 * hdg_score + 0.3 * bear_score + 0.2 * dist_score
            if score > best_score:
                best_score = score
                best = flight
        return best if best_score >= 0.25 else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class FlightTrackerFactory:
    @staticmethod
    def create(provider: str, **kwargs):
        if provider == "flightradar24":
            return FlightRadar24Tracker(**kwargs)
        elif provider == "opensky":
            return OpenSkyTracker(**kwargs)
        else:
            raise ValueError(f"Unknown flight tracker provider: {provider}")


# ---------------------------------------------------------------------------
# Geo helpers (local to this module to avoid circular imports)
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
