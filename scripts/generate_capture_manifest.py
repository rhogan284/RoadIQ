import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2

VIDEO_PATH = "data/road_footage.mp4"
GEOJSON_PATH = Path("src/edgecv/roads/lackey_road.geojson")
OUTPUT_MANIFEST = Path("data/rdd2022/manifest.json")
FRAMES_DIR = Path("data/frames")


def haversine_distance(coord1, coord2):
    R = 6371000  # meters
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def interpolate_position(coords, progress):
    if progress <= 0:
        return coords[0][0], coords[0][1], 0.0
    if progress >= 1:
        return coords[-1][0], coords[-1][1], 0.0

    distances = [haversine_distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    total_dist = sum(distances)
    target_dist = progress * total_dist
    accumulated = 0.0

    for i, d in enumerate(distances):
        if accumulated + d >= target_dist:
            segment_progress = (target_dist - accumulated) / d if d > 0 else 0
            lon = coords[i][0] + segment_progress * (coords[i + 1][0] - coords[i][0])
            lat = coords[i][1] + segment_progress * (coords[i + 1][1] - coords[i][1])

            dlon = math.radians(coords[i + 1][0] - coords[i][0])
            lat1, lat2 = math.radians(coords[i][1]), math.radians(coords[i + 1][1])
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            heading = (math.degrees(math.atan2(y, x)) + 360) % 360
            return lon, lat, heading
        accumulated += d

    return coords[-1][0], coords[-1][1], 0.0


def generate_manifest():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with open(GEOJSON_PATH, "r") as f:
        geojson_data = json.load(f)

    # Directly pull the coordinates from the main feature in lackey_road.geojson
    coords = geojson_data["features"][0]["geometry"]["coordinates"]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    speed_mps = 11.11  # ~40 km/h
    run_id = str(uuid.uuid4())
    device_boot_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)

    manifest_records = []
    seq = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"frame_{seq:04d}.jpg"
        frame_path = FRAMES_DIR / frame_filename

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            seq += 1
            continue

        jpeg_bytes = buffer.tobytes()
        frame_path.write_bytes(jpeg_bytes)
        sha256_hash = hashlib.sha256(jpeg_bytes).hexdigest()

        seconds_elapsed = seq / fps
        frame_time = start_time + timedelta(seconds=seconds_elapsed)
        mono_ns = int(seconds_elapsed * 1e9)

        progress = seq / (total_frames - 1) if total_frames > 1 else 0
        lon, lat, heading = interpolate_position(coords, progress)

        record = {
            "path": f"data/frames/{frame_filename}",
            "run_id": run_id,
            "seq": seq,
            "captured_at": frame_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "capture_mono_ns": mono_ns,
            "device_boot_id": device_boot_id,
            "sha256": sha256_hash,
            "lat": lat,
            "lon": lon,
            "heading_deg": heading,
            "speed_mps": speed_mps,
            "gps_accuracy_m": 3.0,
        }
        manifest_records.append(record)
        seq += 1

    cap.release()

    manifest_data = {"clean": manifest_records, "defect": []}
    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Generated {len(manifest_records)} entries in {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    generate_manifest()