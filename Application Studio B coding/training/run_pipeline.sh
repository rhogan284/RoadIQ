#!/bin/bash
# Waits for the RDD2022 download (curl PID passed as $1) to finish, then runs the full
# prepare -> train -> benchmark pipeline. Logs to training/pipeline.log.
set -e
CURL_PID="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "$CURL_PID" ]; then
  while kill -0 "$CURL_PID" 2>/dev/null; do
    sleep 60
  done
fi

cd "$ROOT"
echo "=== $(date) : download finished, preparing dataset ==="
PYTHONPATH=. python training/prepare_dataset.py

echo "=== $(date) : training YOLO11n ==="
PYTHONPATH=. python training/train_yolo11n.py

echo "=== $(date) : running benchmark (yolo11n + yolov4-tiny) ==="
PYTHONPATH=. python eval/benchmark.py --models yolo11n yolov4-tiny --limit 500

echo "=== $(date) : pipeline complete ==="
