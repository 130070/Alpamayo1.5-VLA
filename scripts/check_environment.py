#!/usr/bin/env python3
"""Check whether the current machine is ready for the Alpamayo video demos."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_PACKAGES = {
    "accelerate": "1.12.0",
    "av": "16.0.1",
    "einops": "0.8.1",
    "huggingface-hub": "0.36.0",
    "hydra-core": "1.3.2",
    "matplotlib": "3.10.7",
    "mediapy": "1.2.4",
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "physical-ai-av": "0.2.0",
    "pillow": "12.0.0",
    "safetensors": "0.7.0",
    "scipy": "1.16.3",
    "seaborn": "0.13.2",
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.57.1",
}

CORE_EXACT_PACKAGES = {"physical-ai-av", "torch", "transformers"}
TESTED_UV_VERSION = "0.11.31"
EXPECTED_UPSTREAM_COMMIT = "f42e594aaf8b50dcd2cbb359d62e3ffc7b12fcf8"
REQUIRED_PROJECT_FILES = (
    Path("pyproject.toml"),
    Path("src/alpamayo1_5/test_inference.py"),
    Path("alpamayo_10s_sliding_demo.py"),
    Path("alpamayo_three_scenes_and_grid.py"),
)
GATED_REPOSITORIES = (
    ("model", "nvidia/Alpamayo-1.5-10B"),
    ("model", "nvidia/Cosmos-Reason2-8B"),
    ("dataset", "nvidia/PhysicalAI-Autonomous-Vehicles"),
)
CACHE_DIRECTORIES = (
    "models--nvidia--Alpamayo-1.5-10B",
    "models--nvidia--Cosmos-Reason2-8B",
    "datasets--nvidia--PhysicalAI-Autonomous-Vehicles",
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


def run_command(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str] | None:
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


def combined_output(result: subprocess.CompletedProcess[str] | None) -> str:
    if result is None:
        return ""
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "no output")


def check_python(reporter: Reporter) -> None:
    version = platform.python_version()
    if sys.version_info[:2] == (3, 12):
        reporter.pass_("Python", f"{version} at {sys.executable}")
    else:
        reporter.fail("Python", f"found {version}; Alpamayo requires Python 3.12.x")

    if sys.prefix != sys.base_prefix:
        reporter.pass_("Virtual environment", sys.prefix)
    else:
        reporter.warn(
            "Virtual environment",
            "not detected; activate a1_5_venv before installing or running the demos",
        )

    system = platform.system()
    if system == "Linux":
        reporter.pass_("Operating system", platform.platform())
    else:
        reporter.warn(
            "Operating system",
            f"{platform.platform()}; final CUDA inference is documented and tested on Linux",
        )


def check_project_layout(reporter: Reporter) -> None:
    root = Path.cwd()
    missing = [str(path) for path in REQUIRED_PROJECT_FILES if not (root / path).exists()]
    if missing:
        reporter.fail(
            "Working directory",
            "run this checker from the official alpamayo1.5 root after copying all VLA scripts; "
            f"missing: {', '.join(missing)}",
        )
    else:
        reporter.pass_("Working directory", f"official project and VLA scripts found in {root}")

    result = run_command(["git", "rev-parse", "HEAD"])
    current_commit = combined_output(result)
    if result is None or result.returncode != 0:
        reporter.fail("Upstream commit", "cannot read the official repository commit")
    elif current_commit == EXPECTED_UPSTREAM_COMMIT:
        reporter.pass_("Upstream commit", current_commit)
    else:
        reporter.warn(
            "Upstream commit",
            f"found {current_commit}; reproduction target is {EXPECTED_UPSTREAM_COMMIT}",
        )


def check_basic_tools(reporter: Reporter) -> None:
    tool_commands = {
        "git": ["git", "--version"],
        "uv": ["uv", "--version"],
        "nvidia-smi": [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        "ffmpeg": ["ffmpeg", "-version"],
        "ffprobe": ["ffprobe", "-version"],
        "hf": ["hf", "--help"],
    }
    for name, command in tool_commands.items():
        if shutil.which(name) is None:
            reporter.fail("Command", f"{name} is not available on PATH")
            continue
        result = run_command(command)
        output = combined_output(result)
        if result is not None and result.returncode == 0:
            line = first_line(output)
            if name == "uv" and TESTED_UV_VERSION not in line:
                reporter.warn(
                    "Command uv",
                    f"{line}; the verified run used uv {TESTED_UV_VERSION}",
                )
            else:
                reporter.pass_(f"Command {name}", line)
        else:
            reporter.fail(f"Command {name}", first_line(output))


def check_packages(reporter: Reporter) -> None:
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            reporter.fail("Python package", f"{distribution} is not installed; expected {expected}")
            continue

        comparable_actual = actual.split("+", maxsplit=1)[0]
        if comparable_actual == expected:
            reporter.pass_("Python package", f"{distribution}=={actual}")
        elif distribution in CORE_EXACT_PACKAGES:
            reporter.fail(
                "Python package",
                f"{distribution}=={actual}; the verified upstream lock uses {expected}",
            )
        else:
            reporter.warn(
                "Python package",
                f"{distribution}=={actual}; the verified upstream lock uses {expected}",
            )

    if importlib.util.find_spec("alpamayo1_5") is None:
        reporter.fail(
            "Alpamayo import",
            "alpamayo1_5 is not importable; run uv sync --active --no-install-package flash-attn",
        )
    else:
        reporter.pass_("Alpamayo import", "alpamayo1_5 is importable")

    try:
        flash_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        reporter.info(
            "FlashAttention",
            "not installed; the repository scripts intentionally use SDPA",
        )
    else:
        reporter.info(
            "FlashAttention",
            f"flash-attn=={flash_version} is installed but not required",
        )


def check_torch_cuda(reporter: Reporter) -> None:
    try:
        import torch
    except Exception as exc:
        reporter.fail("PyTorch runtime", f"cannot import torch: {type(exc).__name__}: {exc}")
        return

    reporter.info(
        "PyTorch runtime",
        f"torch.__version__={torch.__version__}, compiled CUDA={torch.version.cuda}",
    )
    if not torch.cuda.is_available():
        reporter.fail("CUDA runtime", "torch.cuda.is_available() returned False")
        return

    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    reporter.pass_("CUDA runtime", f"available through PyTorch; devices={devices}")
    try:
        value = (torch.tensor([1.0, 2.0], device="cuda") * 2).cpu().tolist()
        reporter.pass_("CUDA calculation", f"test tensor result={value}")
    except Exception as exc:
        reporter.fail("CUDA calculation", f"{type(exc).__name__}: {exc}")


def check_gpu(reporter: Reporter, minimum_free_mib: int) -> None:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        reporter.fail("GPU inventory", first_line(combined_output(result)))
        return

    enough_memory = False
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 6:
            reporter.warn("GPU inventory", f"could not parse nvidia-smi row: {row}")
            continue
        index, name, driver, total, free, utilization = fields
        try:
            free_mib = int(float(free))
        except ValueError:
            free_mib = 0
        enough_memory = enough_memory or free_mib >= minimum_free_mib
        reporter.info(
            f"GPU {index}",
            f"{name}; driver={driver}; total={total} MiB; free={free} MiB; util={utilization}%",
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    reporter.info("CUDA_VISIBLE_DEVICES", visible)
    if enough_memory:
        reporter.pass_("GPU free memory", f"at least one GPU has {minimum_free_mib} MiB free")
    else:
        reporter.fail(
            "GPU free memory",
            f"no GPU has the demo guard of {minimum_free_mib} MiB free",
        )


def check_disk(reporter: Reporter, minimum_free_gib: int) -> None:
    usage = shutil.disk_usage(Path.cwd())
    free_gib = usage.free / 1024**3
    total_gib = usage.total / 1024**3
    detail = f"{free_gib:.1f} GiB free of {total_gib:.1f} GiB on {Path.cwd().anchor or '/'}"
    if free_gib >= minimum_free_gib:
        reporter.pass_("Disk space", detail)
    else:
        reporter.fail("Disk space", f"{detail}; at least {minimum_free_gib} GiB is recommended")


def check_ffmpeg_features(reporter: Reporter) -> None:
    filters = combined_output(run_command(["ffmpeg", "-hide_banner", "-filters"]))
    encoders = combined_output(run_command(["ffmpeg", "-hide_banner", "-encoders"]))
    for feature in ("drawtext", "xstack"):
        if feature in filters:
            reporter.pass_("FFmpeg filter", feature)
        else:
            reporter.fail("FFmpeg filter", f"{feature} is missing")
    if "libx264" in encoders:
        reporter.pass_("FFmpeg encoder", "libx264")
    else:
        reporter.fail("FFmpeg encoder", "libx264 is missing")


def hugging_face_cache_root() -> Path:
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_hub:
        return Path(explicit_hub).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def check_hugging_face_login(reporter: Reporter) -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    reporter.info("Hugging Face endpoint", endpoint)
    result = run_command(["hf", "auth", "whoami"], timeout=30)
    output = combined_output(result)
    if result is not None and result.returncode == 0:
        reporter.pass_("Hugging Face login", first_line(output))
    else:
        reporter.fail("Hugging Face login", f"hf auth whoami failed: {first_line(output)}")

    cache_root = hugging_face_cache_root()
    reporter.info("Hugging Face cache", str(cache_root))
    for directory in CACHE_DIRECTORIES:
        path = cache_root / directory
        if path.exists():
            reporter.pass_("Hugging Face cache entry", directory)
        else:
            reporter.warn("Hugging Face cache entry", f"{directory} is not downloaded yet")


def check_gated_access(reporter: Reporter) -> None:
    try:
        from huggingface_hub import HfApi, get_token
    except Exception as exc:
        reporter.fail("Hugging Face access", f"cannot import huggingface_hub: {exc}")
        return

    token = get_token()
    if not token:
        reporter.fail("Hugging Face access", "no locally saved token was found")
        return

    endpoint = os.environ.get("HF_ENDPOINT") or None
    api = HfApi(endpoint=endpoint, token=token)
    for repo_type, repo_id in GATED_REPOSITORIES:
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, timeout=30)
        except Exception as exc:
            detail = first_line(str(exc)).replace(token, "<redacted>")
            reporter.fail("Gated repository", f"{repo_id}: {type(exc).__name__}: {detail}")
        else:
            reporter.pass_("Gated repository", f"authorized: {repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the official Alpamayo environment before running the VLA demos."
    )
    parser.add_argument(
        "--check-hf-access",
        action="store_true",
        help="query all three gated Hugging Face repositories without downloading them",
    )
    parser.add_argument(
        "--minimum-free-gpu-mib",
        type=int,
        default=30_000,
        help="required free GPU memory; default: 30000 MiB",
    )
    parser.add_argument(
        "--minimum-free-disk-gib",
        type=int,
        default=80,
        help="recommended free disk space; default: 80 GiB",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter()

    print("Alpamayo 1.5 VLA environment check")
    print("=" * 40)
    check_python(reporter)
    check_project_layout(reporter)
    check_disk(reporter, args.minimum_free_disk_gib)
    check_basic_tools(reporter)
    check_packages(reporter)
    check_torch_cuda(reporter)
    check_gpu(reporter, args.minimum_free_gpu_mib)
    check_ffmpeg_features(reporter)
    check_hugging_face_login(reporter)
    if args.check_hf_access:
        check_gated_access(reporter)

    print("=" * 40)
    print(f"Summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
    if reporter.failures:
        print("Environment is not ready. Fix every FAIL item before running inference.")
        return 1
    print("Environment is ready for the repository demos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
