import gc
import json
import subprocess
from pathlib import Path

import physical_ai_av
import torch

import alpamayo_10s_sliding_demo as demo
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


SCENES = [
    {
        "slug": "left_turn_18m",
        "label": "Left turn - 18m",
        "clip_id": "c9c045a3-ebe9-4569-9ce3-a44068cf2e3b",
        "start_us": 2_000_000,
        "nav_text": "Turn left in 18m",
    },
    {
        "slug": "right_turn_14m",
        "label": "Right turn - 14m",
        "clip_id": "82f9f689-6efb-4eb3-af10-070b52eeca22",
        "start_us": 4_000_000,
        "nav_text": "Turn right in 14m",
    },
    {
        "slug": "left_turn_5m",
        "label": "Left turn - 5m",
        "clip_id": "b7dd3c46-ca94-4ffa-a8cf-3a13350c8aba",
        "start_us": 5_000_000,
        "nav_text": "Turn left in 5m",
    },
]

GRID_DIR = Path("outputs/alpamayo_2x2")
GRID_VIDEO = GRID_DIR / "alpamayo_2x2_10s.mp4"
GRID_POSTER = GRID_DIR / "alpamayo_2x2_10s_poster.png"
GRID_META = GRID_DIR / "alpamayo_2x2_meta.json"
ORIGINAL_VIDEO = Path("outputs/alpamayo_10s/alpamayo_10s_result.mp4")


def configure_scene(scene: dict) -> None:
    out_dir = Path("outputs") / f"alpamayo_10s_{scene['slug']}"
    demo.CLIP_ID = scene["clip_id"]
    demo.SCENE_LABEL = scene["label"]
    demo.NAV_TEXT = scene["nav_text"]
    demo.SEGMENT_START_US = scene["start_us"]
    demo.INFERENCE_T0_US = [scene["start_us"] + i * 2_000_000 for i in range(5)]
    demo.OUT_DIR = out_dir
    demo.CACHE_PATH = out_dir / "sliding_results.pt"
    demo.INPUT_VIDEO_PATH = out_dir / "input.mp4"
    demo.RESULT_VIDEO_PATH = out_dir / "result.mp4"
    demo.POSTER_PATH = out_dir / "poster.png"
    demo.META_PATH = out_dir / "meta.json"


def load_shared_model():
    free_mib, util = demo.query_gpu()
    print(f"GPU before shared model load: free={free_mib} MiB util={util}%", flush=True)
    if free_mib is not None and free_mib < demo.REQUIRED_FREE_MIB:
        raise RuntimeError(
            f"Only {free_mib} MiB free; refusing shared model load below "
            f"{demo.REQUIRED_FREE_MIB} MiB"
        )
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    return model, helper.get_processor(model.tokenizer)


def stitch_grid(scene_videos: list[Path]) -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    inputs = [ORIGINAL_VIDEO, *scene_videos]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    filters = []
    labels = [
        "A  Obstacle avoidance",
        "B  Left turn 18m",
        "C  Right turn 14m",
        "D  Left turn 5m",
    ]
    for i, label in enumerate(labels):
        filters.append(
            f"[{i}:v]scale=960:540:flags=lanczos,"
            f"drawtext=text='{label}':x=24:y=22:fontsize=30:"
            "fontcolor=white:box=1:boxcolor=black@0.70:boxborderw=10"
            f"[v{i}]"
        )
    filters.append(
        "[v0][v1][v2][v3]xstack=inputs=4:"
        "layout=0_0|960_0|0_540|960_540[out]"
    )
    command = ["ffmpeg", "-y"]
    for path in inputs:
        command += ["-i", str(path)]
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-t",
        "10",
        "-r",
        str(demo.OUTPUT_FPS),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(GRID_VIDEO),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "8",
            "-i",
            str(GRID_VIDEO),
            "-frames:v",
            "1",
            str(GRID_POSTER),
        ],
        check=True,
    )

    GRID_META.write_text(
        json.dumps(
            {
                "layout": "2x2 synchronized",
                "duration_s": 10,
                "resolution": "1920x1080",
                "fps": demo.OUTPUT_FPS,
                "videos": [str(path) for path in inputs],
                "output": str(GRID_VIDEO),
                "poster": str(GRID_POSTER),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote 2x2 video: {GRID_VIDEO}", flush=True)


def main() -> int:
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    model, processor = load_shared_model()
    caches = []
    try:
        for scene in SCENES:
            configure_scene(scene)
            print(f"\n=== Inference: {scene['label']} ===", flush=True)
            caches.append(demo.run_sliding_inference(model=model, processor=processor, avdi=avdi))
    finally:
        del model, processor
        torch.cuda.empty_cache()
        gc.collect()
        free_mib, util = demo.query_gpu()
        print(f"GPU after inference release: free={free_mib} MiB util={util}%", flush=True)

    scene_videos = []
    for scene, cache in zip(SCENES, caches):
        configure_scene(scene)
        print(f"\n=== Rendering: {scene['label']} ===", flush=True)
        demo.render_videos(cache)
        scene_videos.append(demo.RESULT_VIDEO_PATH)

    stitch_grid(scene_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
