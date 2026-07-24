# VisionPilot Integration on the A100

This document integrates the verified VisionPilot reproduction into the Alpamayo project without
mixing two incompatible model implementations into one Python environment.

## Integration Scope

The two backends solve related driving-perception tasks but have different contracts:

| Backend | Runtime | Input | Main output |
|---|---|---|---|
| Alpamayo 1.5 | Python, PyTorch, SDPA | Four cameras, ego history, navigation text | Future trajectory and Chain-of-Causation text |
| VisionPilot | C++, ONNX Runtime, Docker CUDA | One front camera and vehicle speed | Path, CIPO distance, steering, acceleration, and telemetry |

They therefore remain isolated at runtime:

```text
~/alpamayo1.5/          official NVIDIA Alpamayo source and a1_5_venv
~/Alpamayo1.5-VLA/      shared orchestration, checks, docs, and showcase scripts
~/vision_pilot/         pinned VisionPilot source, Docker context, inputs, and outputs
```

The integration layer provides common A100 checks, repeatable deployment, output validation, and a
labeled side-by-side showcase. The showcase is qualitative because the two systems use different
datasets and camera inputs. It is not a benchmark comparison or model-level sensor fusion.

## Migrated VisionPilot Source

The migration target is the verified repository and commit from the earlier server:

```text
Repository: https://github.com/130070/vision_pilot
Commit:     e8cc95f4ff148aa020ece297141d6ab0a85fa243
Release:    https://github.com/130070/vision_pilot/releases/tag/v1.1
```

That commit contains:

- The full upstream-derived VisionPilot C++ source tree
- Six ONNX models: AutoDrive, AutoSteer, and AutoSpeed in FP32 and INT8 variants
- CUDA, headless, MP4 export, and CSV telemetry changes
- Per-dataset `H.yaml` mounting
- A sample OpenLane input and the existing 2x2 output videos
- Custom-video dataset preparation, frame extraction, and homography generation tools
- English reproduction documentation and CI configuration

The `v1.1` Release archive is a custom CUDA 13 build. Its CUDA provider links to
`libcublas.so.13` and `libcudart.so.13`, so it cannot run with the A100 host's driver 535. The A100
integration instead downloads Microsoft's official
[`onnxruntime-linux-x64-gpu-1.22.0.tgz`](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0),
which targets CUDA 12 and cuDNN 9. The archive is validated before the Docker build and is not
committed into this repository.

The original pinned VisionPilot worktree is kept unchanged. The setup script copies
[`docker/visionpilot-cuda12.Dockerfile`](../docker/visionpilot-cuda12.Dockerfile) into an ignored
`VisionPilot/docker/Dockerfile.cuda12` overlay and stores the compatible runtime as the ignored
`VisionPilot/docker/ort.cuda12.tgz`. This makes the compatibility layer explicit and repeatable
without pretending it belongs to the verified VisionPilot commit.

The old server also contained `output/verify_custom_workflow.mp4` and
`output/verify_custom_workflow.csv` as untracked verification artifacts. They are regenerated on the
A100 so that the new files prove the A100 deployment rather than merely copying an old result.

## A100 Prerequisites

Before deployment, verify:

- Ubuntu Linux
- NVIDIA A100 visible through `nvidia-smi`
- At least 30 GiB free disk space in the user's home filesystem
- Docker Engine and NVIDIA Container Toolkit
- Non-interactive Docker access through the `docker` group or passwordless `sudo docker`
- NVIDIA driver 525 or newer for CUDA 12 minor-version compatibility
- Existing Alpamayo installation under `~/alpamayo1.5`
- This repository under `~/Alpamayo1.5-VLA`

The verified target has an NVIDIA A100-SXM4-80GB and driver `535.183.01`. The integration image uses
CUDA `12.8.1` on Ubuntu 24.04 and ONNX Runtime `1.22.0`. CUDA 12 minor-version compatibility permits
this on driver 535. CUDA 13 is intentionally not used because it would require driver 580 or newer.

The default VisionPilot configuration selects `engine.provider = cuda`. TensorRT packages are
therefore omitted from the compatibility image; this reduces dependency risk and does not change
the provider used by the verified demo.

## 1. Deploy VisionPilot

From the integration repository on the A100:

```bash
cd "$HOME/Alpamayo1.5-VLA"
chmod +x scripts/setup_visionpilot_a100.sh
scripts/setup_visionpilot_a100.sh
```

The setup script:

1. Checks Git, Docker, NVIDIA, disk, and the CUDA 12 driver prerequisite.
2. Clones or validates `~/vision_pilot`.
3. Refuses to overwrite a dirty existing checkout.
4. Checks out the pinned detached commit.
5. Copies the versioned CUDA 12 Dockerfile overlay without modifying the pinned source files.
6. Downloads and validates the official ONNX Runtime CUDA 12 archive.
7. Rebuilds `visionpilot:gpu` for the A100 host.
8. Records source/runtime labels in the Docker image and runs `nvidia-smi` inside it.

To migrate files first without building:

```bash
scripts/setup_visionpilot_a100.sh --skip-build
```

## 2. Check Both Backends

```bash
cd "$HOME/Alpamayo1.5-VLA"
python scripts/check_unified_environment.py
```

The checker verifies both repository layouts, the pinned VisionPilot commit, all six ONNX files,
the ONNX Runtime archive, Docker image, container GPU access, A100 inventory, FFmpeg features,
Alpamayo output, VisionPilot sample input, and both showcase videos.

## 3. Run the VisionPilot A100 Verification

```bash
cd "$HOME/Alpamayo1.5-VLA"
chmod +x scripts/run_visionpilot_demo.sh
scripts/run_visionpilot_demo.sh
```

Expected new A100 artifacts:

```text
~/vision_pilot/output/a100_openlane_verify.mp4
~/vision_pilot/output/a100_openlane_verify.csv
~/vision_pilot/output/a100_openlane_verify.log
```

The CSV header includes frame ID, ego speed, cross-track error, yaw, curvature, CIPO distance,
steering, acceleration, and inference latency for AutoDrive, AutoSteer, and AutoSpeed.

To run a custom dataset:

```bash
scripts/run_visionpilot_demo.sh \
  --dataset "$HOME/vision_pilot/input/custom_phone_video" \
  --output-prefix "$HOME/vision_pilot/output/custom_phone_video_a100"
```

## 4. Create the Unified Showcase

After the Alpamayo and VisionPilot videos exist:

```bash
cd "$HOME/Alpamayo1.5-VLA"
chmod +x scripts/create_unified_showcase.sh
scripts/create_unified_showcase.sh
```

Default output:

```text
~/Alpamayo1.5-VLA/outputs/unified/alpamayo_visionpilot_showcase.mp4
```

The left panel is labeled `Alpamayo 1.5 VLA - multi-camera`; the right panel is labeled
`VisionPilot - single-camera ONNX`. FFmpeg ends at the shorter input instead of looping or inventing
frames.

## 5. Custom VisionPilot Video Workflow

Prepare a phone or dashboard-camera video in the VisionPilot repository:

```bash
cd "$HOME/vision_pilot"

python3 tools/prepare_custom_video_dataset.py \
  /path/to/source.mp4 \
  input/custom_phone_video \
  --speed 0.0 \
  --overwrite

python3 tools/extract_calibration_frame.py \
  input/custom_phone_video/input.mp4 \
  input/custom_phone_video/calibration_frame.jpg \
  --time 2.0
```

Select `near-left`, `near-right`, `far-left`, and `far-right` image points, then generate the
dataset-specific homography:

```bash
python3 tools/generate_h_yaml_from_points.py \
  --image-point 420,690 \
  --image-point 880,690 \
  --image-point 585,430 \
  --image-point 735,430 \
  --lane-width 3.6 \
  --near-distance 6.0 \
  --far-distance 30.0 \
  --output input/custom_phone_video/H.yaml \
  --preview-image input/custom_phone_video/calibration_frame.jpg \
  --preview-output input/custom_phone_video/calibration_preview.jpg
```

The four numbers above are examples, not universal calibration points. Use points measured from the
actual camera frame. VisionPilot automatically mounts the dataset's `H.yaml` when the demo runs.

## Security and Ownership

- No SSH password, private key, GitHub token, or Hugging Face token belongs in either repository.
- Do not stop unrelated A100 processes to free memory.
- Do not commit generated model caches, Docker archives, or gated Alpamayo data.
- Preserve the original VisionPilot and Alpamayo licenses and upstream attribution.
- Change any server password that has previously been exposed in chat or logs.
