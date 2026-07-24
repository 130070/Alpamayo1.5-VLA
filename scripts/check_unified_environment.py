#!/usr/bin/env python3
"""Check an A100 host for both Alpamayo and VisionPilot backends."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


VISIONPILOT_COMMIT = "e8cc95f4ff148aa020ece297141d6ab0a85fa243"
MIN_CUDA12_DRIVER = 525
VISIONPILOT_WEIGHTS = (
    "autodrive_fp32.onnx",
    "autodrive_int8.onnx",
    "autospeed_fp32.onnx",
    "autospeed_int8.onnx",
    "autosteer_fp32.onnx",
    "autosteer_int8.onnx",
)


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def emit(self, status: str, label: str, detail: str) -> None:
        print(f"[{status:<4}] {label}: {detail}")
        if status == "FAIL":
            self.failures += 1
        elif status == "WARN":
            self.warnings += 1

    def pass_(self, label: str, detail: str) -> None:
        self.emit("PASS", label, detail)

    def info(self, label: str, detail: str) -> None:
        self.emit("INFO", label, detail)

    def warn(self, label: str, detail: str) -> None:
        self.emit("WARN", label, detail)

    def fail(self, label: str, detail: str) -> None:
        self.emit("FAIL", label, detail)


def run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def output(result: subprocess.CompletedProcess[str] | None) -> str:
    if result is None:
        return ""
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def first_line(value: str) -> str:
    return next((line.strip() for line in value.splitlines() if line.strip()), "no output")


def docker_command() -> list[str] | None:
    direct = run(["docker", "info"])
    if direct is not None and direct.returncode == 0:
        return ["docker"]
    sudo = run(["sudo", "-n", "docker", "info"])
    if sudo is not None and sudo.returncode == 0:
        return ["sudo", "-n", "docker"]
    return None


def check_host(reporter: Reporter) -> None:
    if platform.system() == "Linux":
        reporter.pass_("Operating system", platform.platform())
    else:
        reporter.fail("Operating system", f"Linux is required; found {platform.platform()}")

    usage = shutil.disk_usage(Path.home())
    free_gib = usage.free / 1024**3
    if free_gib >= 30:
        reporter.pass_("Home disk", f"{free_gib:.1f} GiB free")
    else:
        reporter.fail("Home disk", f"{free_gib:.1f} GiB free; at least 30 GiB is required")

    for command in ("git", "nvidia-smi", "ffmpeg", "ffprobe", "docker"):
        if shutil.which(command):
            reporter.pass_("Host command", command)
        else:
            reporter.fail("Host command", f"missing: {command}")


def check_gpu(reporter: Reporter) -> None:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        reporter.fail("GPU inventory", first_line(output(result)))
        return

    driver_majors: list[int] = []
    for row in result.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 6:
            reporter.warn("GPU inventory", f"could not parse: {row}")
            continue
        index, name, driver, total, free, utilization = fields
        reporter.info(
            f"GPU {index}",
            f"{name}; driver={driver}; total={total} MiB; free={free} MiB; util={utilization}%",
        )
        try:
            driver_majors.append(int(driver.split(".", maxsplit=1)[0]))
        except ValueError:
            pass

    if driver_majors and min(driver_majors) >= MIN_CUDA12_DRIVER:
        reporter.pass_(
            "CUDA 12 driver",
            f"driver {MIN_CUDA12_DRIVER} or newer is available for VisionPilot",
        )
    else:
        reporter.fail(
            "CUDA 12 driver",
            f"the integration image requires driver {MIN_CUDA12_DRIVER} or newer",
        )


def check_alpamayo(reporter: Reporter, root: Path, integration_root: Path) -> None:
    required = (
        root / "pyproject.toml",
        root / "src/alpamayo1_5/test_inference.py",
        root / "a1_5_venv/bin/python",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        reporter.fail("Alpamayo installation", f"missing: {', '.join(missing)}")
    else:
        reporter.pass_("Alpamayo installation", str(root))

    integration_files = (
        integration_root / "scripts/alpamayo_10s_sliding_demo.py",
        integration_root / "scripts/alpamayo_three_scenes_and_grid.py",
        integration_root / "scripts/setup_visionpilot_a100.sh",
    )
    missing = [str(path) for path in integration_files if not path.exists()]
    if missing:
        reporter.fail("Integration repository", f"missing: {', '.join(missing)}")
    else:
        reporter.pass_("Integration repository", str(integration_root))

    videos = (
        root / "outputs/alpamayo_2x2/alpamayo_2x2_10s.mp4",
        integration_root / "assets/videos/alpamayo_2x2_10s.mp4",
    )
    video = next((path for path in videos if path.exists()), None)
    if video is not None:
        reporter.pass_("Alpamayo showcase", str(video))
    else:
        reporter.warn("Alpamayo showcase", f"not found in: {', '.join(map(str, videos))}")


def check_visionpilot(reporter: Reporter, root: Path) -> None:
    required = (
        root / "README.md",
        root / "VisionPilot/docker/run.sh",
        root / "VisionPilot/docker/build.sh",
        root / "VisionPilot/app/vision_pilot.cpp",
        root / "tools/prepare_custom_video_dataset.py",
        root / "tools/extract_calibration_frame.py",
        root / "tools/generate_h_yaml_from_points.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        reporter.fail("VisionPilot installation", f"missing: {', '.join(missing)}")
    else:
        reporter.pass_("VisionPilot installation", str(root))

    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    actual_commit = output(commit)
    if commit is None or commit.returncode != 0:
        reporter.fail("VisionPilot commit", first_line(actual_commit))
    elif actual_commit == VISIONPILOT_COMMIT:
        reporter.pass_("VisionPilot commit", actual_commit)
    else:
        reporter.warn(
            "VisionPilot commit",
            f"found {actual_commit}; migration target is {VISIONPILOT_COMMIT}",
        )

    weight_root = root / "VisionPilot/modules/models/weights"
    missing_weights = [name for name in VISIONPILOT_WEIGHTS if not (weight_root / name).is_file()]
    if missing_weights:
        reporter.fail("VisionPilot weights", f"missing: {', '.join(missing_weights)}")
    else:
        reporter.pass_("VisionPilot weights", "all six FP32/INT8 ONNX files are present")

    ort_archive = root / "VisionPilot/docker/ort.cuda12.tgz"
    if ort_archive.is_file():
        archive = run(["tar", "-tzf", str(ort_archive)], timeout=60)
        if archive is not None and archive.returncode == 0:
            reporter.pass_("ONNX Runtime archive", str(ort_archive))
        else:
            reporter.fail("ONNX Runtime archive", f"invalid archive: {ort_archive}")
    else:
        reporter.fail("ONNX Runtime archive", f"missing: {ort_archive}")

    sample = root / "input/openlane_sample/input.mp4"
    if sample.exists():
        reporter.pass_("VisionPilot sample", str(sample))
    else:
        reporter.fail("VisionPilot sample", f"missing: {sample}")

    showcase = root / "output/openlane_2x2_grid.mp4"
    if showcase.exists():
        reporter.pass_("VisionPilot showcase", str(showcase))
    else:
        reporter.warn("VisionPilot showcase", f"not present: {showcase}")


def check_docker(reporter: Reporter) -> None:
    docker = docker_command()
    if docker is None:
        reporter.fail("Docker access", "docker info failed for the user and passwordless sudo")
        return
    reporter.pass_("Docker access", " ".join(docker))

    image = run([*docker, "image", "inspect", "visionpilot:gpu"])
    if image is None or image.returncode != 0:
        reporter.fail("VisionPilot image", "visionpilot:gpu is not built")
        return
    reporter.pass_("VisionPilot image", "visionpilot:gpu")

    gpu = run(
        [
            *docker,
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "nvidia-smi",
            "visionpilot:gpu",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    if gpu is not None and gpu.returncode == 0:
        reporter.pass_("Container GPU", first_line(gpu.stdout))
    else:
        reporter.fail("Container GPU", first_line(output(gpu)))


def check_ffmpeg(reporter: Reporter) -> None:
    filters = output(run(["ffmpeg", "-hide_banner", "-filters"]))
    encoders = output(run(["ffmpeg", "-hide_banner", "-encoders"]))
    for feature in ("drawtext", "hstack", "xstack"):
        if feature in filters:
            reporter.pass_("FFmpeg filter", feature)
        else:
            reporter.fail("FFmpeg filter", f"missing: {feature}")
    if "libx264" in encoders:
        reporter.pass_("FFmpeg encoder", "libx264")
    else:
        reporter.fail("FFmpeg encoder", "missing: libx264")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpamayo-root",
        type=Path,
        default=Path(os.environ.get("ALPAMAYO_ROOT", "~/alpamayo1.5")).expanduser(),
    )
    parser.add_argument(
        "--visionpilot-root",
        type=Path,
        default=Path(os.environ.get("VISIONPILOT_ROOT", "~/vision_pilot")).expanduser(),
    )
    parser.add_argument(
        "--integration-root",
        type=Path,
        default=Path(os.environ.get("INTEGRATION_ROOT", "~/Alpamayo1.5-VLA")).expanduser(),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter()
    print("Alpamayo + VisionPilot unified A100 check")
    print("=" * 46)
    check_host(reporter)
    check_gpu(reporter)
    check_alpamayo(reporter, args.alpamayo_root.resolve(), args.integration_root.resolve())
    check_visionpilot(reporter, args.visionpilot_root.resolve())
    check_docker(reporter)
    check_ffmpeg(reporter)
    print("=" * 46)
    print(f"Summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
    return 1 if reporter.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
