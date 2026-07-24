#!/usr/bin/env bash
set -euo pipefail

VISIONPILOT_ROOT="${VISIONPILOT_ROOT:-$HOME/vision_pilot}"
DATASET=""
OUTPUT_PREFIX=""

usage() {
    cat <<'EOF'
Run one VisionPilot video dataset with GPU inference, MP4 export, and CSV telemetry.

Usage:
  run_visionpilot_demo.sh [--root PATH] [--dataset PATH] [--output-prefix PATH]

Defaults:
  root:          ~/vision_pilot
  dataset:       <root>/input/openlane_sample
  output-prefix: <root>/output/a100_openlane_verify
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)
            [ "$#" -ge 2 ] || { echo "error: --root requires a path" >&2; exit 2; }
            VISIONPILOT_ROOT="$2"
            shift 2
            ;;
        --dataset)
            [ "$#" -ge 2 ] || { echo "error: --dataset requires a path" >&2; exit 2; }
            DATASET="$2"
            shift 2
            ;;
        --output-prefix)
            [ "$#" -ge 2 ] || { echo "error: --output-prefix requires a path" >&2; exit 2; }
            OUTPUT_PREFIX="$2"
            shift 2
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

VISIONPILOT_ROOT="$(realpath -m "$VISIONPILOT_ROOT")"
DATASET="$(realpath -m "${DATASET:-$VISIONPILOT_ROOT/input/openlane_sample}")"
OUTPUT_PREFIX="$(realpath -m "${OUTPUT_PREFIX:-$VISIONPILOT_ROOT/output/a100_openlane_verify}")"

for input in "$DATASET/input.mp4" "$DATASET/frame_speed.txt"; do
    [ -f "$input" ] || { echo "error: required dataset file is missing: $input" >&2; exit 1; }
done

run_script="$VISIONPILOT_ROOT/VisionPilot/docker/run.sh"
[ -x "$run_script" ] || { echo "error: run script is missing or not executable: $run_script" >&2; exit 1; }

if docker info >/dev/null 2>&1; then
    PRIVILEGE=()
elif sudo -n docker info >/dev/null 2>&1; then
    PRIVILEGE=(sudo -n)
else
    echo "error: Docker requires interactive sudo or the current user lacks Docker access." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
output_video="${OUTPUT_PREFIX}.mp4"
output_csv="${OUTPUT_PREFIX}.csv"
output_log="${OUTPUT_PREFIX}.log"

echo "VisionPilot A100 verification"
echo "Dataset: $DATASET"
echo "Video:   $output_video"
echo "CSV:     $output_csv"

(
    cd "$VISIONPILOT_ROOT/VisionPilot/docker"
    "${PRIVILEGE[@]}" ./run.sh --gpu --no-display \
        --data "$DATASET:/data" \
        --output-video "$output_video" \
        --output-csv "$output_csv"
) 2>&1 | tee "$output_log"

if [ "${#PRIVILEGE[@]}" -gt 0 ]; then
    "${PRIVILEGE[@]}" chown "$(id -u):$(id -g)" "$output_video" "$output_csv" || true
fi

[ -s "$output_video" ] || { echo "error: output video was not created" >&2; exit 1; }
[ -s "$output_csv" ] || { echo "error: telemetry CSV was not created" >&2; exit 1; }

ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
    -of json \
    "$output_video"

header="$(head -1 "$output_csv")"
echo "CSV header: $header"
echo "VisionPilot verification completed successfully."
