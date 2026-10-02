import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
from edgecv.feedsim.route import RouteTrack

VIDEO_PATH = "data/road_footage.mp4"
GEOJSON_PATH = Path("src/edgecv/roads/sydney_demo.geojson")
OUTPUT_MANIFEST = "data/capture_manifest.json"
FRAMES_DIR = Path("data/frames")


def generate_manifest():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return

    # Ensure output directory for extracted frame images exists
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video: {fps:.2f} FPS, {total_frames} total frames.")

    # Speed in metres per second (40 km/h = 11.11 m/s)
    speed_mps = 40.0 / 3.6

    # Instantiate RouteTrack using its valid 'from_network' class method
    track = RouteTrack.from_network(
        path=GEOJSON_PATH,
        mode="random",
        n_frames=total_frames,
        speed_mps=speed_mps,
        fps=fps,
    )

    # Contract 5 mandatory identifiers
    run_id = str(uuid.uuid4())
    device_boot_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)

    manifest_records = []
    seq = 0

    print("Extracting frame metadata and saving images for capture manifest...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save actual image frame to disk
        frame_path = FRAMES_DIR / f"frame_{seq:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # Calculate time offsets
        seconds_elapsed = seq / fps
        frame_time = start_time + timedelta(seconds=seconds_elapsed)
        mono_ns = int(seconds_elapsed * 1e9)

        # Hash JPEG frame buffer for data integrity
        _, buffer = cv2.imencode(".jpg", frame)
        sha256_hash = hashlib.sha256(buffer).hexdigest()

        # Get GPS Fix using RouteTrack's native fix_for method
        fix = track.fix_for(seq)

        # Contract 5 JSON item shape + mandatory "path" key for feedsim
        record = {
            "path": str(frame_path).replace("\\", "/"),
            "run_id": run_id,
            "seq": seq,
            "captured_at": frame_time.isoformat(),
            "capture_mono_ns": mono_ns,
            "device_boot_id": device_boot_id,
            "sha256": sha256_hash,
            "lat": fix.lat,
            "lon": fix.lon,
            "heading_deg": fix.heading_deg,
            "speed_mps": fix.speed_mps,
            "gps_accuracy_m": fix.gps_accuracy_m,
        }

        manifest_records.append(record)
        seq += 1

    cap.release()

    manifest_data = {"clean": manifest_records, "defect": []}

    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(
        f"Done! Generated {len(manifest_records)} entries in {OUTPUT_MANIFEST}"
    )


if __name__ == "__main__":
    generate_manifest()