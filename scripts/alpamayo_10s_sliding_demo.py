import gc
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mediapy as mp
import numpy as np
import physical_ai_av
import torch
from PIL import Image, ImageDraw, ImageFont

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
SCENE_LABEL = "Obstacle avoidance"
NAV_TEXT = None
SEGMENT_START_US = 5_100_000
SEGMENT_DURATION_S = 10.0
INFERENCE_T0_US = [5_100_000, 7_100_000, 9_100_000, 11_100_000, 13_100_000]
OUTPUT_FPS = 5
REQUIRED_FREE_MIB = 30_000

OUT_DIR = Path("outputs/alpamayo_10s")
CACHE_PATH = OUT_DIR / "alpamayo_10s_sliding_results.pt"
INPUT_VIDEO_PATH = OUT_DIR / "alpamayo_10s_input.mp4"
RESULT_VIDEO_PATH = OUT_DIR / "alpamayo_10s_result.mp4"
POSTER_PATH = OUT_DIR / "alpamayo_10s_result_poster.png"
META_PATH = OUT_DIR / "alpamayo_10s_meta.json"

CAMERA_FEATURES = [
    (0, "cross_left", "camera_cross_left_120fov", (1, 0)),
    (1, "front_wide", "camera_front_wide_120fov", (1, 1)),
    (2, "cross_right", "camera_cross_right_120fov", (1, 2)),
    (6, "front_tele", "camera_front_tele_30fov", (0, 1)),
]


def font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def query_gpu() -> tuple[int | None, int | None]:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        free, util = p.stdout.strip().splitlines()[0].split(",")
        return int(free.strip()), int(util.strip())
    except Exception:
        return None, None


def plain_cot(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, str) and value.startswith("['") and value.endswith("']"):
        value = value[2:-2]
    return str(value)


def min_ade(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    gt_xy = gt_xyz[0, 0, :, :2].numpy().T
    pred_xy_all = pred_xyz.numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    return float(np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=1).mean(-1).min())


def run_sliding_inference(model=None, processor=None, avdi=None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {
        "clip_id": CLIP_ID,
        "segment_start_us": SEGMENT_START_US,
        "segment_duration_s": SEGMENT_DURATION_S,
        "inference_t0_us": INFERENCE_T0_US,
        "navigation_instruction": NAV_TEXT,
        "results": [],
    }
    if CACHE_PATH.exists():
        loaded = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        if (
            loaded.get("clip_id") == CLIP_ID
            and loaded.get("inference_t0_us") == INFERENCE_T0_US
            and loaded.get("navigation_instruction") == NAV_TEXT
        ):
            cache = loaded
            for result in cache["results"]:
                result["cot"] = plain_cot(result["cot"])
            torch.save(cache, CACHE_PATH)
            print(f"Resuming cache with {len(cache['results'])}/{len(INFERENCE_T0_US)} windows", flush=True)

    done = {int(x["t0_us"]) for x in cache["results"]}
    missing = [t for t in INFERENCE_T0_US if t not in done]
    if not missing:
        print("All sliding-window inference results already cached", flush=True)
        return cache

    owns_model = model is None
    if owns_model:
        free_mib, util = query_gpu()
        print(f"GPU before model load: free={free_mib} MiB, util={util}%", flush=True)
        if free_mib is not None and free_mib < REQUIRED_FREE_MIB:
            raise RuntimeError(
                f"Only {free_mib} MiB GPU memory free; refusing to load Alpamayo below "
                f"the {REQUIRED_FREE_MIB} MiB guard."
            )
        print("Loading Alpamayo 1.5 once with SDPA attention", flush=True)
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        processor = helper.get_processor(model.tokenizer)
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    try:
        for index, t0_us in enumerate(INFERENCE_T0_US, start=1):
            if t0_us in done:
                continue
            free_mib, util = query_gpu()
            print(
                f"Window {index}/{len(INFERENCE_T0_US)} t0={t0_us / 1e6:.1f}s "
                f"GPU free={free_mib} MiB util={util}%",
                flush=True,
            )
            if free_mib is not None and free_mib < 8_000:
                raise RuntimeError(f"GPU free memory fell to {free_mib} MiB; stopping before inference")

            data = load_physical_aiavdataset(CLIP_ID, t0_us=t0_us, avdi=avdi)
            messages = helper.create_message(
                frames=data["image_frames"].flatten(0, 1),
                camera_indices=data["camera_indices"],
                nav_text=NAV_TEXT,
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = helper.to_device(
                {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                },
                "cuda",
            )
            torch.cuda.manual_seed_all(42 + index)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )

            pred_cpu = pred_xyz.cpu()
            result = {
                "t0_us": t0_us,
                "history_xyz": data["ego_history_xyz"].cpu(),
                "gt_xyz": data["ego_future_xyz"].cpu(),
                "pred_xyz": pred_cpu,
                "cot": plain_cot(extra["cot"][0]),
                "min_ade_m": min_ade(pred_cpu, data["ego_future_xyz"].cpu()),
            }
            cache["results"].append(result)
            cache["results"].sort(key=lambda x: int(x["t0_us"]))
            torch.save(cache, CACHE_PATH)
            print(
                f"Saved window t0={t0_us / 1e6:.1f}s, minADE={result['min_ade_m']:.4f} m, "
                f"CoC={result['cot']}",
                flush=True,
            )
            del data, messages, inputs, model_inputs, pred_xyz, pred_cpu, extra
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        if owns_model:
            del model, processor
            torch.cuda.empty_cache()
            gc.collect()
    return cache


def label_image(img: Image.Image, text: str, size: int = 22) -> None:
    draw = ImageDraw.Draw(img)
    f = font(size)
    box = draw.textbbox((12, 10), text, font=f)
    draw.rectangle((6, 5, box[2] + 7, box[3] + 5), fill=(0, 0, 0))
    draw.text((12, 10), text, font=f, fill="white")


def load_video_grids() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    timestamps = SEGMENT_START_US + (
        np.arange(int(SEGMENT_DURATION_S * OUTPUT_FPS), dtype=np.int64)
        * int(1_000_000 / OUTPUT_FPS)
    )
    cell_w, cell_h = 420, 236
    resized_by_camera = {}
    front_wide_frames = []
    for cam_id, name, feature, _position in CAMERA_FEATURES:
        print(f"Decoding 10-second input from {name}", flush=True)
        reader = avdi.get_clip_feature(CLIP_ID, feature, maybe_stream=True)
        frames, _actual_timestamps = reader.decode_images_from_timestamps(timestamps)
        reader.close()
        resized = []
        for frame_arr in frames:
            if cam_id == 1:
                front_wide_frames.append(
                    np.asarray(
                        Image.fromarray(frame_arr).resize((1280, 720), Image.Resampling.LANCZOS)
                    )
                )
            img = Image.fromarray(frame_arr).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            label_image(img, name)
            resized.append(np.asarray(img))
        resized_by_camera[cam_id] = resized
        del frames, resized
        gc.collect()

    grids = []
    for i, timestamp_us in enumerate(timestamps):
        grid = Image.new("RGB", (cell_w * 3, cell_h * 2), "black")
        for cam_id, _name, _feature, (row, col) in CAMERA_FEATURES:
            grid.paste(Image.fromarray(resized_by_camera[cam_id][i]), (col * cell_w, row * cell_h))
        draw = ImageDraw.Draw(grid)
        draw.text(
            (12, 12),
            f"dataset t={timestamp_us / 1e6:.1f}s",
            font=font(24),
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        grids.append(np.asarray(grid.resize((1280, 480), Image.Resampling.LANCZOS)))
    return grids, front_wide_frames, timestamps


def wrap_text(draw: ImageDraw.ImageDraw, text: str, f, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=f)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def trajectory_panel(result: dict) -> np.ndarray:
    history = result["history_xyz"][0, 0, :, :2].numpy()
    gt = result["gt_xyz"][0, 0, :, :2].numpy()
    pred = result["pred_xyz"][0, 0, 0, :, :2].numpy()

    fig = plt.figure(figsize=(6.4, 2.4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(history[:, 0], history[:, 1], color="0.5", linewidth=2.0, label="history")
    ax.plot(gt[:, 0], gt[:, 1], color="tab:red", linewidth=2.5, label="ground truth")
    ax.plot(pred[:, 0], pred[:, 1], color="tab:blue", linewidth=2.5, label="prediction")
    ax.scatter([0], [0], marker="^", color="black", s=55, label="ego at t0")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"Prediction at t={result['t0_us'] / 1e6:.1f}s | minADE={result['min_ade_m']:.3f}m",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.18)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return panel


def result_bottom_panel(result: dict) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 240), "white")
    plot = Image.fromarray(trajectory_panel(result)).resize((640, 240), Image.Resampling.LANCZOS)
    canvas.paste(plot, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    body_font = font(18)
    draw.text((660, 20), "Alpamayo 1.5 Chain-of-Causation", font=title_font, fill="black")
    y = 62
    for line in wrap_text(draw, result["cot"], body_font, 590):
        draw.text((660, y), line, font=body_font, fill=(20, 20, 20))
        y += 28
    draw.text(
        (660, 195),
        "Sliding-window inference: prediction refreshes every 2.0 seconds",
        font=font(15),
        fill=(70, 70, 70),
    )
    return np.asarray(canvas)


def load_projection_context() -> dict:
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    extrinsics = avdi.get_clip_feature(
        CLIP_ID,
        avdi.features.CALIBRATION.SENSOR_EXTRINSICS,
        maybe_stream=True,
    )
    intrinsics = avdi.get_clip_feature(
        CLIP_ID,
        avdi.features.CALIBRATION.CAMERA_INTRINSICS,
        maybe_stream=True,
    )
    egomotion = avdi.get_clip_feature(
        CLIP_ID,
        avdi.features.LABELS.EGOMOTION,
        maybe_stream=True,
    )
    camera_name = "camera_front_wide_120fov"
    return {
        "tf_vehicle_camera": extrinsics.sensor_poses[camera_name],
        "camera_model": intrinsics.camera_models[camera_name],
        "egomotion": egomotion,
    }


def transform_from_prediction_frame(
    points_t0: np.ndarray,
    prediction_t0_us: int,
    current_timestamp_us: int,
    context: dict,
) -> np.ndarray:
    ego_t0 = context["egomotion"](prediction_t0_us).pose
    ego_current = context["egomotion"](current_timestamp_us).pose
    tf_current_t0 = ego_current.inv() * ego_t0
    return tf_current_t0.apply(points_t0)


def trajectory_corridor(points: np.ndarray, half_width_m: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(points[:, :2], axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.maximum(tangent_norm, 1e-5)
    normal = np.concatenate([-tangent[:, 1:2], tangent[:, 0:1]], axis=1) / tangent_norm
    left = points.copy()
    right = points.copy()
    left[:, :2] += normal * half_width_m
    right[:, :2] -= normal * half_width_m
    return left, right


def project_vehicle_points(points_vehicle: np.ndarray, context: dict) -> tuple[np.ndarray, np.ndarray]:
    points_camera = context["tf_vehicle_camera"].inv().apply(points_vehicle)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = context["camera_model"].ray2pixel(points_camera)
    valid = (
        (points_camera[:, 2] > 0.5)
        & np.isfinite(pixels).all(axis=1)
        & (~context["camera_model"].is_out_of_bounds(pixels))
    )
    pixels = pixels * np.array([1280 / 1920, 720 / 1080])
    return pixels, valid


def pixel_tuples(pixels: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in pixels[valid]]


def processed_front_frame(
    frame: np.ndarray,
    result: dict,
    timestamp_us: int,
    context: dict,
    mini_plot: np.ndarray,
) -> np.ndarray:
    pred_t0 = result["pred_xyz"][0, 0, 0].numpy()
    gt_t0 = result["gt_xyz"][0, 0].numpy()
    left_t0, right_t0 = trajectory_corridor(pred_t0)

    pred_current = transform_from_prediction_frame(
        pred_t0, int(result["t0_us"]), timestamp_us, context
    )
    gt_current = transform_from_prediction_frame(
        gt_t0, int(result["t0_us"]), timestamp_us, context
    )
    left_current = transform_from_prediction_frame(
        left_t0, int(result["t0_us"]), timestamp_us, context
    )
    right_current = transform_from_prediction_frame(
        right_t0, int(result["t0_us"]), timestamp_us, context
    )

    pred_px, pred_valid = project_vehicle_points(pred_current, context)
    gt_px, gt_valid = project_vehicle_points(gt_current, context)
    left_px, left_valid = project_vehicle_points(left_current, context)
    right_px, right_valid = project_vehicle_points(right_current, context)

    base = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    corridor_valid = pred_valid & left_valid & right_valid
    corridor_indices = np.flatnonzero(corridor_valid)
    if len(corridor_indices) >= 3:
        polygon = [tuple(map(int, left_px[i])) for i in corridor_indices]
        polygon += [tuple(map(int, right_px[i])) for i in corridor_indices[::-1]]
        draw.polygon(polygon, fill=(0, 140, 255, 72))
        draw.line(
            [tuple(map(int, left_px[i])) for i in corridor_indices],
            fill=(40, 180, 255, 205),
            width=3,
        )
        draw.line(
            [tuple(map(int, right_px[i])) for i in corridor_indices],
            fill=(40, 180, 255, 205),
            width=3,
        )

    pred_line = pixel_tuples(pred_px, pred_valid)
    gt_line = pixel_tuples(gt_px, gt_valid)
    if len(pred_line) >= 2:
        draw.line(pred_line, fill=(0, 170, 255, 255), width=8, joint="curve")
        for x, y in pred_line[::8]:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(220, 250, 255, 255))
    if len(gt_line) >= 2:
        draw.line(gt_line, fill=(255, 70, 60, 235), width=5, joint="curve")

    composed = Image.alpha_composite(base, overlay)
    mini = Image.fromarray(mini_plot).resize((390, 160), Image.Resampling.LANCZOS).convert("RGBA")
    mini.putalpha(225)
    composed.alpha_composite(mini, (870, 18))

    info = Image.new("RGBA", composed.size, (0, 0, 0, 0))
    info_draw = ImageDraw.Draw(info)
    info_draw.rounded_rectangle((18, 18, 520, 126), radius=12, fill=(0, 0, 0, 175))
    info_draw.text(
        (36, 32),
        f"{SCENE_LABEL} | prediction @ {result['t0_us'] / 1e6:.1f}s",
        font=font(25),
        fill="white",
    )
    info_draw.rectangle((37, 76, 83, 88), fill=(0, 170, 255, 230))
    info_draw.text((96, 69), "Predicted driving corridor", font=font(18), fill="white")
    info_draw.rectangle((37, 105, 83, 112), fill=(255, 70, 60, 240))
    info_draw.text((96, 94), "Ground-truth future", font=font(18), fill="white")

    info_draw.rectangle((0, 575, 1280, 720), fill=(0, 0, 0, 190))
    info_draw.text((28, 590), "Chain-of-Causation", font=font(22), fill=(130, 210, 255))
    y = 625
    for line in wrap_text(info_draw, result["cot"], font(22), 1200):
        info_draw.text((28, y), line, font=font(22), fill="white")
        y += 31
    info_draw.text(
        (985, 690),
        f"minADE {result['min_ade_m']:.3f} m",
        font=font(17),
        fill=(220, 220, 220),
    )
    return np.asarray(Image.alpha_composite(composed, info).convert("RGB"))


def input_frame(grid: np.ndarray, relative_s: float) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 720), "black")
    canvas.paste(Image.fromarray(grid), (0, 120))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 28), "PhysicalAI-AV 10-second multi-camera input", font=font(30), fill="white")
    draw.text(
        (28, 72),
        f"clip {CLIP_ID} | segment {relative_s:.1f}/{SEGMENT_DURATION_S:.1f}s",
        font=font(20),
        fill=(220, 220, 220),
    )
    return np.asarray(canvas)


def render_videos(cache: dict) -> None:
    grids, front_wide_frames, timestamps = load_video_grids()
    results = sorted(cache["results"], key=lambda x: int(x["t0_us"]))
    if len(results) != len(INFERENCE_T0_US):
        raise RuntimeError(f"Expected {len(INFERENCE_T0_US)} results, found {len(results)}")
    mini_plots = {int(r["t0_us"]): trajectory_panel(r) for r in results}
    projection_context = load_projection_context()

    input_frames, result_frames = [], []
    for i, (grid, front_frame, timestamp_us) in enumerate(
        zip(grids, front_wide_frames, timestamps)
    ):
        relative_s = i / OUTPUT_FPS
        active = max(
            (r for r in results if int(r["t0_us"]) <= int(timestamp_us)),
            key=lambda r: int(r["t0_us"]),
        )
        input_frames.append(input_frame(grid, relative_s))
        result_frames.append(
            processed_front_frame(
                front_frame,
                active,
                int(timestamp_us),
                projection_context,
                mini_plots[int(active["t0_us"])],
            )
        )

    input_video = np.stack(input_frames)
    result_video = np.stack(result_frames)
    mp.write_video(str(INPUT_VIDEO_PATH), input_video, fps=OUTPUT_FPS)
    mp.write_video(str(RESULT_VIDEO_PATH), result_video, fps=OUTPUT_FPS)
    Image.fromarray(result_video[-1]).save(POSTER_PATH)

    metadata = {
        "clip_id": CLIP_ID,
        "scene_label": SCENE_LABEL,
        "navigation_instruction": NAV_TEXT,
        "segment_start_s": SEGMENT_START_US / 1e6,
        "segment_duration_s": SEGMENT_DURATION_S,
        "output_fps": OUTPUT_FPS,
        "output_frames": int(result_video.shape[0]),
        "method": "Alpamayo-supported sliding-window inference over a continuous 10-second segment",
        "model_input_per_window": "1.6 s ego history plus the latest 4 frames from each of 4 cameras",
        "prediction_per_window": "6.4 s future trajectory plus Chain-of-Causation text",
        "prediction_refresh_s": 2.0,
        "visualization": "Camera-calibrated projection on the front-wide image",
        "corridor_note": (
            "The blue driving corridor is derived from the model-predicted center trajectory; "
            "it is not a separately predicted lane-boundary segmentation."
        ),
        "windows": [
            {
                "t0_s": r["t0_us"] / 1e6,
                "min_ade_m": r["min_ade_m"],
                "chain_of_causation": r["cot"],
            }
            for r in results
        ],
        "input_video": str(INPUT_VIDEO_PATH),
        "result_video": str(RESULT_VIDEO_PATH),
        "poster": str(POSTER_PATH),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote input video: {INPUT_VIDEO_PATH}", flush=True)
    print(f"Wrote processed video: {RESULT_VIDEO_PATH}", flush=True)
    print(f"Wrote poster: {POSTER_PATH}", flush=True)
    print(f"Wrote metadata: {META_PATH}", flush=True)


def main() -> int:
    cache = run_sliding_inference()
    render_videos(cache)
    free_mib, util = query_gpu()
    print(f"GPU after completion: free={free_mib} MiB, util={util}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
