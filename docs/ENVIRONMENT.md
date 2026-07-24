# Environment and Version Reference

This document separates the versions observed during the successful reproduction from the
versions resolved by the official `NVlabs/alpamayo1.5` dependency lock. Do not build a second
independent Python environment for this repository. The two demo scripts run inside the official
Alpamayo environment.

The official source and lock target is commit
`f42e594aaf8b50dcd2cbb359d62e3ffc7b12fcf8`. Use:

```bash
git -C "$HOME/alpamayo1.5" checkout f42e594aaf8b50dcd2cbb359d62e3ffc7b12fcf8
```

## Verified Hardware and Runtime

| Component | Verified value | Notes |
|---|---|---|
| Setup GPU | NVIDIA RTX 4060 Ti, 8 GB | Environment setup and downloads succeeded; inference did not fit |
| Final GPU | NVIDIA A100, 80 GB | Completed SDPA inference and video generation |
| Python | `3.12.3` | The official project requires `==3.12.*` |
| uv | `0.11.31` | Used to create and synchronize `a1_5_venv` |
| PyTorch runtime | `2.8.0+cu128` | The `+cu128` suffix means the wheel contains CUDA 12.8 runtime support |
| CUDA reported by PyTorch | `12.8` | This is separate from the NVIDIA driver version |
| Attention implementation | `sdpa` | FlashAttention was intentionally skipped |
| Alpamayo model cache | approximately 21-22 GB | Stored in the Hugging Face cache |
| Additional model VRAM | approximately 21.8 GB | Observed with one trajectory sample |
| Demo memory guard | 30,000 MiB free | Checked before loading the model |

The NVIDIA driver and FFmpeg build can vary by server. The environment checker prints their
actual versions and verifies the required FFmpeg filters and H.264 encoder.

## Python Dependency Versions

The official project directly pins the three core packages shown below. The remaining exact
versions are the resolutions recorded in the official `uv.lock` used as the reproduction target.
Running `uv sync --active --no-install-package flash-attn` installs this set into the active
environment.

| Distribution | Reproduction version |
|---|---:|
| `accelerate` | `1.12.0` |
| `av` | `16.0.1` |
| `einops` | `0.8.1` |
| `huggingface-hub` | `0.36.0` |
| `hydra-core` | `1.3.2` |
| `matplotlib` | `3.10.7` |
| `mediapy` | `1.2.4` |
| `numpy` | `2.3.5` |
| `pandas` | `2.3.3` |
| `physical-ai-av` | `0.2.0` |
| `pillow` | `12.0.0` |
| `safetensors` | `0.7.0` |
| `scipy` | `1.16.3` |
| `seaborn` | `0.13.2` |
| `torch` | `2.8.0` |
| `torchvision` | `0.23.0` |
| `transformers` | `4.57.1` |

`flash-attn` is not part of the tested environment. Both repository demo scripts explicitly pass
`attn_implementation="sdpa"` when loading Alpamayo.

## Run the Environment Checker

After copying the scripts into the official repository root and activating `a1_5_venv`:

```bash
cd "$HOME/alpamayo1.5"
source a1_5_venv/bin/activate
mkdir -p outputs
set -o pipefail
python check_environment.py --check-hf-access | tee outputs/environment-check.txt
```

The checker does not download model weights or dataset archives. It verifies:

- Linux, Python 3.12, the active virtual environment, and project layout
- Free disk space
- `git`, `uv`, `nvidia-smi`, `ffmpeg`, `ffprobe`, and `hf`
- Every Python package in the table above
- PyTorch CUDA availability and a small CUDA tensor calculation
- GPU name, driver, utilization, total VRAM, and currently free VRAM
- FFmpeg `drawtext`, `xstack`, and `libx264` support
- Hugging Face login, endpoint, cache entries, and all three gated repositories

`PASS` means the check matched the reproduction target. `WARN` is informational or indicates a
non-core version difference. Every `FAIL` should be fixed before inference. The process exits with
code `1` when any failure exists.

To change only the resource thresholds:

```bash
python check_environment.py \
  --check-hf-access \
  --minimum-free-gpu-mib 30000 \
  --minimum-free-disk-gib 80
```

Do not lower the GPU threshold merely to force a run on an 8 GB or 16 GB GPU.

## Save a Complete Environment Record

Run these commands after a successful check:

```bash
mkdir -p outputs/environment
uv pip freeze > outputs/environment/python-packages.txt
nvidia-smi > outputs/environment/nvidia-smi.txt
ffmpeg -version > outputs/environment/ffmpeg-version.txt 2>&1
git rev-parse HEAD > outputs/environment/upstream-commit.txt
git -C "$HOME/Alpamayo1.5-VLA" rev-parse HEAD \
  > outputs/environment/vla-commit.txt
```

These text files make it possible to compare a later machine with the environment that produced
its videos. They do not contain model weights or Hugging Face tokens.
