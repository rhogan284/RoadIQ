import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
from edgecv.feedsim.route import RouteTrack

VIDEO_PATH = "data/road_footage.mp4"
GEOJSON_PATH = Path("src/edgecv/roads/sydney_demo.geojson")

# Save directly to the target location feedsim uses
OUTPUT_MANIFEST = Path("data/rdd2022/manifest.json")
FRAMES_DIR = Path("data/frames")


def generate_manifest():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video: {fps:.2f} FPS, {total_frames} total frames.")

    speed_mps = 40.0 / 3.6

    track = RouteTrack.from_network(
        path=GEOJSON_PATH,
        mode="random",
        n_frames=total_frames,
        speed_mps=speed_mps,
        fps=fps,
    )

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

        frame_filename = f"frame_{seq:04d}.jpg"
        frame_path = FRAMES_DIR / frame_filename

        # Encode JPEG once to guarantee file on disk and hash match exactly
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

        fix = track.fix_for(seq)

        record = {
            # Use POSIX path relative to repository root
            "path": f"data/frames/{frame_filename}",
            "run_id": run_id,
            "seq": seq,
            "captured_at": frame_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
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

    print(f"Done! Generated {len(manifest_records)} entries in {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    generate_manifest()