# Build context is the parent directory (`Application Studio B coding/`), not this folder —
# see docker-compose.yml's `build.context: ..`. That's what lets this Dockerfile COPY
# requirements.txt / detector_interface / eval / training / examples directly.
#
# Python pinned to 3.13 to match the exact environment this project is already proven to run on
# (verified 2026-08-22): numpy==2.5.1 has no wheel for 3.11 at all (requires-python >=3.12) —
# confirmed against PyPI directly, not assumed. Python 3.12 was never actually tested against this
# exact dependency set, whereas 3.13 is what's running successfully on the host today, hence 3.13
# here rather than 3.12. (onnx was also suspected to need 3.12 over 3.13 at design time — no wheel
# is tagged literally cp313 — but the actual build below shows that's not a real constraint: pip
# resolves onnx's cp312-abi3 wheel as forward-compatible with 3.13, so this never came up.)
FROM python:3.13-slim

# git        — `pip install git+https://...` (YOLOX) shells out to a real git binary.
# curl/ca-certificates — fetch the pinned NanoDet source tarball below.
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1 — required for `opencv-python` (non-headless;
#   YOLOv4-tiny needs cv2.dnn.readNetFromDarknet, which the headless build doesn't ship) to import.
# build-essential/cmake — kept as a source-build fallback for onnx, based on PyPI metadata that
#   suggested no cp313 wheel existed. The actual build proved that wrong: pip resolves onnx's
#   cp312-abi3 wheel (forward-compatible with 3.13, same mechanism as opencv-python below), so no
#   source build ever happens and these two packages currently do nothing at image build time.
#   Left in rather than removed, since it's a correctness-neutral safety net if a future onnx pin
#   ever does need to build from source — but don't assume they're load-bearing today.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch, explicit. The default PyPI index for torch==2.13.0 bundles CUDA 13.0 +
# cuDNN (every wheel listed there is a GPU build, ~500MB plus several GB of nvidia-* transitive
# deps) — verified via PyPI, not assumed. This host has no GPU, so pull the CPU-only build from
# PyTorch's dedicated CPU wheel index instead, matching the `torch==2.13.0+cpu` already installed
# on the host. requirements.txt's own `torch==2.13.0` pin is satisfied by this and won't be
# refetched from the default index when installed below.
#
# torchvision installed from the same CPU index in the same step, for the same reason — caught by
# actually running a smoke test, not assumed up front: ultralytics pulls in torchvision as a
# dependency, and left unpinned here it resolves torchvision==0.28.0 from the *default* PyPI
# index (matching version number, but a different compiled build than the `+cpu` torch above),
# which crashes every torchvision-dependent import with
# `RuntimeError: operator torchvision::nms does not exist` — a binary ABI mismatch between the
# two builds, not a version mismatch. Pinning it here to the CPU index's matching build fixes
# that at the source instead of pinning a workaround-version downstream.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLOX-Tiny / YOLOX-Nano: the `yolox` PyPI sdist is broken (missing requirements.txt upstream),
# so it's installed from GitHub source instead. --no-deps because yolox's own requirements.txt
# pulls onnx-simplifier, torchvision, psutil etc. that aren't needed for the detect()-only code
# path these two plugins actually exercise — installing exactly the runtime deps below instead
# keeps the image reproducibly matching what's proven to work on the host, rather than whatever
# yolox's full dependency set happens to resolve to.
RUN pip install --no-cache-dir loguru tabulate thop ninja tqdm tensorboard pycocotools onnx \
    && pip install --no-cache-dir --no-build-isolation --no-deps \
        git+https://github.com/Megvii-BaseDetection/YOLOX.git

# NanoDet-m: no usable PyPI/GitHub-installable package — upstream's nanodet/model/ and
# nanodet/data/ have no __init__.py, so setuptools silently drops them from any built wheel.
# Vendored here as a pinned-commit tarball (not a git clone) so every build gets exactly this
# source, byte-for-byte, without needing git history in the image or relying on a teammate having
# already cloned it locally (third_party/nanodet is gitignored — a fresh clone has nothing there).
# Commit be9b4a9 (2024-01-08) is the exact commit already vendored and proven working against
# nanodet_m.py's torch._six shim and config/legacy_v0.x_configs/nanodet-m.yml path.
RUN curl -fsSL \
        https://github.com/RangiLyu/nanodet/archive/be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0.tar.gz \
        -o /tmp/nanodet.tar.gz \
    && mkdir -p third_party \
    && tar xzf /tmp/nanodet.tar.gz -C third_party \
    && mv third_party/nanodet-be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0 third_party/nanodet \
    && rm /tmp/nanodet.tar.gz
# nanodet's own utils import both of these; requirements.txt doesn't pin exact versions since
# they're only needed for this one plugin (matches the host's own Setup instructions).
#
# pytorch_lightning pinned <2.0.0 — third_party/nanodet/requirements.txt itself pins
# `pytorch-lightning>=1.9.0,<2.0.0`, but an unpinned `pip install pytorch_lightning` resolves the
# latest (2.6.5 as of 2026-08-23) regardless. PL 2.0 removed the `training_epoch_end` hook
# nanodet's TrainingTask relies on — training crashes immediately with a NotImplementedError on
# the unpinned version. Confirmed by actually hitting this on the host and fixing it there first.
#
# imagesize: nanodet/data/dataset/yolo.py (the built-in YOLO-format dataset loader,
# training/train_nanodet_m.py's route to fine-tuning this model) imports it directly, but it's
# not pulled in by anything else — missing on the host until training/train_nanodet_m.py was
# first written and run, and not listed in nanodet's own requirements.txt either.
RUN pip install --no-cache-dir "pytorch_lightning<2.0.0" termcolor imagesize

COPY detector_interface/ ./detector_interface/
COPY eval/benchmark.py ./eval/benchmark.py
COPY training/ ./training/
COPY examples/ ./examples/

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# Persistent dev container — `docker compose exec detector <script>` runs any training/benchmark/
# example script on demand, rather than this image having one fixed purpose.
CMD ["sleep", "infinity"]
