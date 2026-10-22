"""
Dual-head aircraft type + airline livery classifier.

Architecture: EfficientNetV2-M backbone (from timm) with two separate
classification heads:
  1. Aircraft type head  (100 FGVC-Aircraft classes)
  2. Airline livery head (60 major airline classes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label lists
# ---------------------------------------------------------------------------

AIRCRAFT_TYPE_LABELS: List[str] = [
    "707-320", "727-200", "737-200", "737-300", "737-400", "737-500",
    "737-600", "737-700", "737-800", "737-900", "747-100", "747-200",
    "747-300", "747-400", "757-200", "757-300", "767-200", "767-300",
    "767-400", "777-200", "777-300", "787-8", "787-9", "787-10",
    "A300B4", "A310", "A318", "A319", "A320", "A321", "A330-200",
    "A330-300", "A340-200", "A340-300", "A340-500", "A340-600",
    "A350-900", "A350-1000", "A380", "A220-100", "A220-300",
    "ATR-42", "ATR-72", "An-12",
    "BAE 146-200", "BAE 146-300", "BAE-125",
    "Beechcraft 1900",
    "Boeing 717",
    "C-130", "C-47",
    "CRJ-200", "CRJ-700", "CRJ-900",
    "Cessna 172", "Cessna 208", "Cessna Citation",
    "Challenger 600",
    "DC-10", "DC-3", "DC-6", "DC-8", "DC-9-30",
    "DHC-1", "DHC-6",
    "Dash 8 Q400",
    "Dornier 328",
    "E-170", "E-175", "E-190", "E-195",
    "EMB-120",
    "ERJ-135", "ERJ-145",
    "Embraer Legacy 600",
    "Eurofighter Typhoon",
    "F/A-18",
    "Fokker 100", "Fokker 50", "Fokker 70",
    "Global Express",
    "Gulfstream IV", "Gulfstream V",
    "Hawk T1",
    "IL-76",
    "King Air 200",
    "L-1011",
    "MD-11", "MD-80", "MD-87", "MD-90",
    "Metroliner",
    "Model B200",
    "PA-28",
    "SR-20",
    "Saab 2000", "Saab 340",
    "Spitfire",
    "Tornado",
    "Tristar",
    "Tu-134", "Tu-154",
    "Yak-42",
    "737-MAX7", "737-MAX8", "737-MAX9", "737-MAX10",
    "A321neo", "A321XLR", "A330neo",
]

# Pad / trim to exactly 100
while len(AIRCRAFT_TYPE_LABELS) < 100:
    AIRCRAFT_TYPE_LABELS.append(f"unknown_{len(AIRCRAFT_TYPE_LABELS)}")
AIRCRAFT_TYPE_LABELS = AIRCRAFT_TYPE_LABELS[:100]

AIRLINE_LABELS: List[str] = [
    "Air Canada", "Air China", "Air France", "Air India", "Air New Zealand",
    "Alaska Airlines", "Alitalia", "All Nippon Airways", "American Airlines",
    "Asiana Airlines", "Austrian Airlines", "Avianca",
    "British Airways", "Brussels Airlines",
    "Cathay Pacific", "China Eastern", "China Southern",
    "Delta Air Lines",
    "EasyJet", "Emirates", "Ethiopian Airlines", "Etihad Airways",
    "EVA Air",
    "Finnair",
    "Flydubai",
    "Gulf Air",
    "Hainan Airlines",
    "Iberia", "IndiGo",
    "Japan Airlines", "Jeju Air",
    "Kenya Airways", "Korean Air",
    "LATAM Airlines", "Lion Air", "Lufthansa",
    "Malaysia Airlines",
    "Norwegian Air",
    "Philippine Airlines",
    "Qatar Airways", "Qantas",
    "Royal Jordanian",
    "Ryanair",
    "SAS Scandinavian", "Singapore Airlines", "South African Airways",
    "Southwest Airlines", "SpiceJet", "Spirit Airlines",
    "Swiss International",
    "Thai Airways", "Turkish Airlines",
    "United Airlines",
    "Vietnam Airlines", "Virgin Atlantic", "Virgin Australia",
    "Vueling",
    "WestJet", "Wizz Air",
    "XiamenAir",
]

while len(AIRLINE_LABELS) < 60:
    AIRLINE_LABELS.append(f"unknown_airline_{len(AIRLINE_LABELS)}")
AIRLINE_LABELS = AIRLINE_LABELS[:60]


# ---------------------------------------------------------------------------
# Variant database
# ---------------------------------------------------------------------------

class AircraftVariantDB:
    """Maps base aircraft type to known variants."""

    _DB: Dict[str, List[str]] = {
        "A320": ["A320ceo", "A320neo", "A320neo ACF"],
        "A321": ["A321ceo", "A321neo", "A321XLR", "A321LR"],
        "A319": ["A319ceo", "A319neo"],
        "A318": ["A318"],
        "A330": ["A330-200", "A330-300", "A330-200F", "A330neo"],
        "A350": ["A350-900", "A350-1000", "A350F"],
        "A380": ["A380-800"],
        "A220": ["A220-100", "A220-300"],
        "B737": ["B737-700", "B737-800", "B737-900", "B737 MAX 7", "B737 MAX 8",
                 "B737 MAX 9", "B737 MAX 10"],
        "B747": ["B747-400", "B747-8I", "B747-8F"],
        "B757": ["B757-200", "B757-300"],
        "B767": ["B767-200", "B767-300", "B767-300F", "B767-400"],
        "B777": ["B777-200", "B777-200ER", "B777-200LR", "B777-300", "B777-300ER",
                 "B777X", "B777-8", "B777-9"],
        "B787": ["B787-8", "B787-9", "B787-10"],
        "E-Jets": ["E170", "E175", "E190", "E195", "E175-E2", "E190-E2", "E195-E2"],
        "CRJ": ["CRJ-200", "CRJ-700", "CRJ-900", "CRJ-1000"],
        "ATR": ["ATR-42-500", "ATR-42-600", "ATR-72-500", "ATR-72-600"],
    }

    @classmethod
    def get_variants(cls, base_type: str) -> List[str]:
        for key, variants in cls._DB.items():
            if key.lower() in base_type.lower() or base_type.lower() in key.lower():
                return variants
        return [base_type]

    @classmethod
    def infer_variant(
        cls,
        aircraft_type: str,
        airline: str,
        visual_features: Optional[Dict] = None,
    ) -> str:
        """
        Infer the most likely variant from aircraft type, airline, and optional
        visual feature hints (e.g. winglet presence detected by secondary model).
        """
        variants = cls.get_variants(aircraft_type)
        if not variants:
            return aircraft_type

        # Simple heuristic: prefer newer variants for newer airlines
        neo_airlines = {
            "IndiGo", "Wizz Air", "Norwegian Air", "Vueling", "EasyJet",
            "Air France", "Lufthansa", "Singapore Airlines",
        }
        if any(a.lower() in airline.lower() for a in neo_airlines):
            neo_variants = [v for v in variants if "neo" in v.lower() or "max" in v.lower()]
            if neo_variants:
                return neo_variants[0]

        return variants[0]


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    aircraft_type: str
    variant: str
    airline: str
    confidence: float
    top3: List[Tuple[str, float]] = field(default_factory=list)
    top3_airlines: List[Tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_dual_head_model(num_aircraft: int, num_airline: int):
    """Build EfficientNetV2-M dual-head classifier using timm."""
    try:
        import timm
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("timm and torch are required") from exc

    class DualHeadClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnetv2_m",
                pretrained=True,
                num_classes=0,  # remove original head
            )
            feat_dim = self.backbone.num_features
            self.aircraft_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_aircraft),
            )
            self.airline_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_airline),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.aircraft_head(features), self.airline_head(features)

    return DualHeadClassifier()


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class AircraftClassifier:
    """
    Classifies an aircraft crop into type + airline.

    Can operate without trained weights (returns 'unknown' with 0 confidence)
    so the rest of the pipeline does not crash during development.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_aircraft_classes: int = 100,
        num_airline_classes: int = 60,
        device: str = "auto",
        image_size: int = 224,
        top_k: int = 3,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.num_aircraft_classes = num_aircraft_classes
        self.num_airline_classes = num_airline_classes
        self.image_size = image_size
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.variant_db = AircraftVariantDB()

        self.device = self._resolve_device(device)
        self.model = None
        self.transform = None

        self._setup_transforms()

        if model_path and Path(model_path).exists():
            self.load_model(model_path, num_aircraft_classes, num_airline_classes)
        else:
            logger.warning(
                "No classifier weights found at %s. "
                "classify() will return 'unknown' until weights are trained.",
                model_path,
            )

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

    def _setup_transforms(self) -> None:
        try:
            import torchvision.transforms as T
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        except ImportError:
            logger.warning("torchvision not available; preprocessing disabled.")
            self.transform = None

    def load_model(
        self,
        model_path: str,
        num_aircraft_classes: int,
        num_airline_classes: int,
    ) -> None:
        """Load trained weights from checkpoint file."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required") from exc

        self.model = _build_dual_head_model(num_aircraft_classes, num_airline_classes)
        state = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Classifier loaded from %s on %s", model_path, self.device)

    # ------------------------------------------------------------------

    def preprocess(self, crop: np.ndarray):
        """Convert a BGR numpy crop to a normalised tensor batch."""
        import cv2 as _cv2
        rgb = _cv2.cvtColor(crop, _cv2.COLOR_BGR2RGB)
        if self.transform is None:
            raise RuntimeError("torchvision transforms not initialised.")
        tensor = self.transform(rgb)
        return tensor.unsqueeze(0).to(self.device)

    def classify(self, frame_crop: np.ndarray) -> ClassificationResult:
        """
        Classify an aircraft image crop.

        Returns ClassificationResult with type/variant/airline predictions.
        Falls back to 'unknown' if model is not loaded or crop is invalid.
        """
        _UNKNOWN = ClassificationResult(
            aircraft_type="unknown",
            variant="unknown",
            airline="unknown",
            confidence=0.0,
            top3=[("unknown", 0.0)],
            top3_airlines=[("unknown", 0.0)],
        )

        if frame_crop is None or frame_crop.size == 0:
            return _UNKNOWN

        if self.model is None:
            return _UNKNOWN

        try:
            import torch
            import torch.nn.functional as F

            tensor = self.preprocess(frame_crop)

            with torch.no_grad():
                type_logits, airline_logits = self.model(tensor)
                type_probs = F.softmax(type_logits, dim=-1)[0].cpu().numpy()
                airline_probs = F.softmax(airline_logits, dim=-1)[0].cpu().numpy()

            # Top-K aircraft types
            k = min(self.top_k, len(AIRCRAFT_TYPE_LABELS))
            top_type_idx = np.argsort(type_probs)[::-1][:k]
            top3 = [(AIRCRAFT_TYPE_LABELS[i], float(type_probs[i])) for i in top_type_idx]

            # Top-K airlines
            k_air = min(self.top_k, len(AIRLINE_LABELS))
            top_air_idx = np.argsort(airline_probs)[::-1][:k_air]
            top3_airlines = [(AIRLINE_LABELS[i], float(airline_probs[i])) for i in top_air_idx]

            best_type, best_type_conf = top3[0]
            best_airline, _ = top3_airlines[0]

            variant = self.variant_db.infer_variant(best_type, best_airline)

            return ClassificationResult(
                aircraft_type=best_type,
                variant=variant,
                airline=best_airline,
                confidence=best_type_conf,
                top3=top3,
                top3_airlines=top3_airlines,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Classification error: %s", exc)
            return _UNKNOWN
