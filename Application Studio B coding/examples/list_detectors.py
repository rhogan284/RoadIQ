"""Demonstrates the registry without needing any weights downloaded or dependencies installed."""

from detector_interface import AVAILABLE_DETECTORS, get_detector

for name in AVAILABLE_DETECTORS:
    detector = get_detector(name)
    print(f"{detector.name} (version={detector.version})")
    for key, value in detector.params.items():
        print(f"  {key}: {value}")
