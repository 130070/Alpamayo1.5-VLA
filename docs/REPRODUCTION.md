# Reproduction Guide

This document describes the exact execution structure used to produce the videos in this repository.

## 1. Environment

The final inference environment was created under the upstream Alpamayo 1.5 repository with Python 3.12. PyTorch SDPA was used instead of FlashAttention because the initial environment did not include a compatible `flash-attn` build.

The scripts expect these packages to be available:

```text
torch
transformers
physical_ai_av
numpy
scipy
Pillow
matplotlib
mediapy
```

FFmpeg must be discoverable through `PATH` for MP4 encoding and the 2x2 composite.

The complete tested version matrix is recorded in [ENVIRONMENT.md](ENVIRONMENT.md). After cloning
both repositories, copy all three utility files and run the checker from the official project root:

```bash
cp "$HOME/Alpamayo1.5-VLA/scripts/"*.py "$HOME/alpamayo1.5/"
cd "$HOME/alpamayo1.5"
source a1_5_venv/bin/activate
python check_environment.py --check-hf-access
```

The checker is separate from the two demo programs. It performs no model inference and downloads no
weights.

## 2. Access Requirements

The model and dataset may require accepting Hugging Face access conditions. Authenticate using the standard Hugging Face CLI or the authentication method documented by the upstream project. Do not put access tokens in source files, shell history, commits, or issue reports.

An optional mirror was used in the tested environment:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

This variable is not required when direct Hugging Face access is reliable.

## 3. Input Structure

For each reference timestamp, `load_physical_aiavdataset` loads:

- 16 ego-history states at 10 Hz, ending at the reference time
- 64 future ground-truth states at 10 Hz
- Four recent frames from each selected camera
- Dataset timestamps and calibration data

The selected camera indices are `[0, 1, 2, 6]`, corresponding to:

```text
0 -> camera_cross_left_120fov
1 -> camera_front_wide_120fov
2 -> camera_cross_right_120fov
6 -> camera_front_tele_30fov
```

## 4. Model Invocation

The model is loaded with:

```python
model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

The processor message combines camera frames, ego history, and an optional navigation instruction. Sampling uses one trajectory per window:

```python
pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
    data=model_inputs,
    top_p=0.98,
    temperature=0.6,
    num_traj_samples=1,
    max_generation_length=256,
    return_extra=True,
)
```

The cache stores only CPU tensors and plain text, so rendering can be repeated without loading the model again.

## 5. Temporal Alignment

A prediction generated at `t0` is expressed in the ego frame at `t0`. Reusing it on later frames without transforming it would cause a visible projection error. For every video timestamp `t`, the script uses dataset ego-motion poses to convert the prediction through the world frame and into the current ego frame.

The transformation sequence is:

```text
prediction ego(t0) -> world -> ego(t) -> front camera(t) -> image pixels
```

Points behind the camera, non-finite projections, and pixels outside image bounds are removed.

## 6. Corridor Construction

The model returns a center trajectory. The renderer estimates a local tangent at each waypoint, computes a perpendicular normal in the ego `x-y` plane, and offsets the center point by `+/-1.4 m`. The resulting left and right curves form the translucent blue polygon.

This corridor is a presentation layer. It is not an additional neural-network output and must not be used as a lane-boundary estimate.

## 7. Single-Scene Execution

Copy all files from this repository's `scripts` directory into the upstream repository root, run
`check_environment.py`, and review the constants at the top of
`alpamayo_10s_sliding_demo.py`:

```python
CLIP_ID
SCENE_LABEL
NAV_TEXT
SEGMENT_START_US
INFERENCE_T0_US
OUTPUT_FPS
REQUIRED_FREE_MIB
```

Then run:

```bash
PYTHONUNBUFFERED=1 python alpamayo_10s_sliding_demo.py
```

If the run is interrupted after any completed inference window, the `.pt` cache preserves that window. Re-running the same configuration resumes missing windows and then renders the videos.

## 8. Batch Execution

Copy both Python scripts into the upstream project root and run:

```bash
PYTHONUNBUFFERED=1 python alpamayo_three_scenes_and_grid.py
```

The batch configuration is defined by the `SCENES` list. Each item contains:

```python
{
    "slug": "left_turn_18m",
    "label": "Left turn - 18m",
    "clip_id": "...",
    "start_us": 2_000_000,
    "nav_text": "Turn left in 18m",
}
```

The shared model remains on the GPU while all missing windows are computed. It is deleted before camera decoding and FFmpeg encoding.

## 9. Output Verification

Use FFprobe to confirm the combined artifact:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
  -of json \
  outputs/alpamayo_2x2/alpamayo_2x2_10s.mp4
```

Expected values:

```json
{
  "codec_name": "h264",
  "width": 1920,
  "height": 1080,
  "r_frame_rate": "5/1",
  "duration": "10.000000",
  "nb_frames": "50"
}
```

## 10. Troubleshooting

### CUDA out of memory

Stop before model loading if the free-memory guard is not satisfied. Do not reduce the guard when another important workload shares the same GPU. Use a dedicated GPU whenever possible.

### Dataset reference lookup fails

Confirm Hugging Face authentication, dataset access approval, and network connectivity. If a mirror is used, confirm that the repository revision and required files exist on that mirror.

### FlashAttention import or initialization failure

Use the tested SDPA fallback:

```python
attn_implementation="sdpa"
```

### The trajectory appears detached from the road

Confirm that projection uses the calibration belonging to the same clip and that the prediction is transformed from `ego(t0)` into `ego(t)` before camera projection.

### FFmpeg `xstack` rejects `fill`

Older FFmpeg releases may support `xstack` but not its `fill` option. Remove `:fill=black` when all four scaled inputs already fill the complete 2x2 canvas.

## 11. Reproducibility Notes

- Seeds are set for each inference window.
- Exact language output can still depend on software versions and model implementation details.
- Cached results should be deleted after changing clip IDs, navigation instructions, model settings, or sampling parameters unless the cache-validation fields are updated accordingly.
- Quantitative comparisons should use a broader evaluation set and multiple trajectory samples.
