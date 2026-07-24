#!/usr/bin/env bash
set -euo pipefail

VISIONPILOT_URL="https://github.com/130070/vision_pilot.git"
VISIONPILOT_REF="e8cc95f4ff148aa020ece297141d6ab0a85fa243"
VISIONPILOT_ROOT="${VISIONPILOT_ROOT:-$HOME/vision_pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKERFILE_SOURCE="$INTEGRATION_ROOT/docker/visionpilot-cuda12.Dockerfile"
ORT_VERSION="1.22.0"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
ORT_SHA256="2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55"
CUDA_TAG="12.8.1-devel-ubuntu24.04"
SKIP_BUILD=0
SKIP_DOWNLOAD=0

usage() {
    cat <<'EOF'
Deploy the pinned VisionPilot reproduction beside Alpamayo on an A100 server.

Usage:
  setup_visionpilot_a100.sh [options]

Options:
  --root PATH       Installation directory. Default: ~/vision_pilot
  --skip-download   Require an existing valid VisionPilot/docker/ort.cuda12.tgz.
  --skip-build      Migrate files but do not build the Docker image.
  -h, --help        Show this help.

This script does not install Docker, NVIDIA drivers, or NVIDIA Container Toolkit. It preserves the
pinned VisionPilot worktree and builds with a separate CUDA 12 Dockerfile and ONNX Runtime archive.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)
            [ "$#" -ge 2 ] || { echo "error: --root requires a path" >&2; exit 2; }
            VISIONPILOT_ROOT="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for command in git curl tar nvidia-smi sha256sum; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "error: required command is missing: $command" >&2
        exit 1
    }
done

if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
    PRIVILEGE=()
elif sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo -n docker)
    PRIVILEGE=(sudo -n)
else
    echo "error: Docker is unavailable or the current user lacks non-interactive access." >&2
    echo "Add the user to the docker group or configure passwordless sudo for Docker." >&2
    exit 1
fi

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
driver_major="${driver_version%%.*}"
if ! [[ "$driver_major" =~ ^[0-9]+$ ]]; then
    echo "error: could not parse NVIDIA driver version: $driver_version" >&2
    exit 1
fi

free_kib="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
free_gib="$((free_kib / 1024 / 1024))"
if [ "$free_gib" -lt 30 ]; then
    echo "error: only ${free_gib} GiB is free under $HOME; at least 30 GiB is required." >&2
    exit 1
fi

echo "VisionPilot A100 deployment"
echo "============================"
echo "Target root:  $VISIONPILOT_ROOT"
echo "Pinned ref:   $VISIONPILOT_REF"
echo "Driver:       $driver_version"
echo "Free disk:    ${free_gib} GiB"
echo "CUDA image:   $CUDA_TAG"
echo "ONNX Runtime: $ORT_VERSION official CUDA 12 package"

[ -f "$DOCKERFILE_SOURCE" ] || {
    echo "error: integration Dockerfile is missing: $DOCKERFILE_SOURCE" >&2
    exit 1
}

if [ -e "$VISIONPILOT_ROOT" ] && [ ! -d "$VISIONPILOT_ROOT/.git" ]; then
    echo "error: target exists but is not a Git repository: $VISIONPILOT_ROOT" >&2
    exit 1
fi

if [ ! -d "$VISIONPILOT_ROOT/.git" ]; then
    git clone "$VISIONPILOT_URL" "$VISIONPILOT_ROOT"
else
    tracked_dirty="$(git -C "$VISIONPILOT_ROOT" status --porcelain --untracked-files=no)"
    unexpected_untracked="$(
        git -C "$VISIONPILOT_ROOT" ls-files --others --exclude-standard |
            grep -Ev '^(output|data|logs)/|^VisionPilot/docker/(Dockerfile\.cuda12|ort\.cuda12\.tgz(\.partial)?)$' || true
    )"
    if [ -n "$tracked_dirty" ] || [ -n "$unexpected_untracked" ]; then
        echo "error: existing VisionPilot checkout contains unexpected changes." >&2
        echo "Only generated files under output/, data/, and logs/ are allowed." >&2
        git -C "$VISIONPILOT_ROOT" status --short >&2
        exit 1
    fi
    remote_url="$(git -C "$VISIONPILOT_ROOT" remote get-url origin)"
    case "$remote_url" in
        *130070/vision_pilot*) ;;
        *)
            echo "error: unexpected VisionPilot origin: $remote_url" >&2
            exit 1
            ;;
    esac
    if git -C "$VISIONPILOT_ROOT" cat-file -e "${VISIONPILOT_REF}^{commit}" 2>/dev/null; then
        echo "Pinned VisionPilot commit is already available locally; skipping fetch."
    else
        git -C "$VISIONPILOT_ROOT" fetch --depth 1 origin "$VISIONPILOT_REF"
    fi
fi

git -C "$VISIONPILOT_ROOT" checkout --detach "$VISIONPILOT_REF"
mkdir -p "$VISIONPILOT_ROOT/output" "$VISIONPILOT_ROOT/data" "$VISIONPILOT_ROOT/logs"

dockerfile_path="$VISIONPILOT_ROOT/VisionPilot/docker/Dockerfile.cuda12"
ort_path="$VISIONPILOT_ROOT/VisionPilot/docker/ort.cuda12.tgz"
cp "$DOCKERFILE_SOURCE" "$dockerfile_path"

info_exclude="$VISIONPILOT_ROOT/.git/info/exclude"
for generated in VisionPilot/docker/Dockerfile.cuda12 VisionPilot/docker/ort.cuda12.tgz; do
    grep -qxF "/$generated" "$info_exclude" || printf '/%s\n' "$generated" >> "$info_exclude"
done

archive_root="onnxruntime-linux-x64-gpu-${ORT_VERSION}"
archive_valid=0
if archive_listing="$(tar -tzf "$ort_path" 2>/dev/null)"; then
    if grep -qx "${archive_root}/lib/libonnxruntime_providers_cuda.so" <<< "$archive_listing"; then
        archive_valid=1
    fi
fi
if [ "$archive_valid" -ne 1 ]; then
    if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
        echo "error: --skip-download was used but $ort_path is missing or invalid." >&2
        exit 1
    fi
    rm -f "${ort_path}.partial"
    download_ok=0
    for attempt in $(seq 1 10); do
        echo "ONNX Runtime download attempt $attempt/10"
        if curl -fL \
            --continue-at - \
            --connect-timeout 30 \
            --output "${ort_path}.partial" \
            "$ORT_URL"; then
            download_ok=1
            break
        fi
        sleep 5
    done
    if [ "$download_ok" -ne 1 ]; then
        echo "error: ONNX Runtime download failed after 10 attempts." >&2
        exit 1
    fi
    tar -tzf "${ort_path}.partial" >/dev/null
    mv "${ort_path}.partial" "$ort_path"
fi
echo "ONNX Runtime archive verified: $ort_path"
actual_ort_sha256="$(sha256sum "$ort_path" | awk '{print $1}')"
if [ "$actual_ort_sha256" != "$ORT_SHA256" ]; then
    echo "error: ONNX Runtime SHA256 mismatch." >&2
    echo "Expected: $ORT_SHA256" >&2
    echo "Actual:   $actual_ort_sha256" >&2
    exit 1
fi
echo "ONNX Runtime SHA256: $actual_ort_sha256"

if [ "$SKIP_BUILD" -eq 0 ]; then
    if [ "$driver_major" -lt 525 ]; then
        echo "error: the CUDA 12 integration image requires driver 525 or newer." >&2
        echo "Files were migrated, but the image was not built. Current driver: $driver_version" >&2
        exit 1
    fi
    (
        cd "$VISIONPILOT_ROOT/VisionPilot/docker"
        "${DOCKER[@]}" build \
            --progress=plain \
            --build-arg "CUDA_TAG=$CUDA_TAG" \
            --build-arg ENABLE_ROS2=OFF \
            --build-arg BUILD_JOBS=8 \
            --label "org.opencontainers.image.revision=$VISIONPILOT_REF" \
            --label "org.opencontainers.image.version=onnxruntime-${ORT_VERSION}-cuda12" \
            -t visionpilot:gpu \
            -f Dockerfile.cuda12 \
            ..
    )
    "${DOCKER[@]}" image inspect visionpilot:gpu >/dev/null
    "${DOCKER[@]}" run --rm --gpus all --entrypoint nvidia-smi visionpilot:gpu \
        --query-gpu=name,memory.total --format=csv,noheader
fi

echo "============================"
echo "VisionPilot migration complete."
echo "Commit: $(git -C "$VISIONPILOT_ROOT" rev-parse HEAD)"
echo "Docker: $dockerfile_path"
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "Image:  visionpilot:gpu"
else
    echo "Image:  skipped"
fi
