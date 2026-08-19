"""Re-export the Detector protocol so detector authors import from one place."""
from edgecv.contracts.detection import (  # noqa: F401
    BBox, Detection, Detector, DetectorInfo, severity_for,
)
