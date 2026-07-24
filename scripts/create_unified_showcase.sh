#!/usr/bin/env bash
set -euo pipefail

ALPAMAYO_VIDEO="${ALPAMAYO_VIDEO:-$HOME/Alpamayo1.5-VLA/assets/videos/alpamayo_2x2_10s.mp4}"
VISIONPILOT_VIDEO="${VISIONPILOT_VIDEO:-$HOME/vision_pilot/output/a100_openlane_verify.mp4}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-$HOME/Alpamayo1.5-VLA/outputs/unified/alpamayo_visionpilot_showcase.mp4}"

usage() {
    cat <<'EOF'
Create a labeled side-by-side showcase of the Alpamayo and VisionPilot results.

Usage:
  create_unified_showcase.sh [--alpamayo VIDEO] [--visionpilot VIDEO] [--output VIDEO]

This is a qualitative showcase. The two backends use different datasets and inputs, so the
result must not be described as a quantitative benchmark comparison.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --alpamayo)
            [ "$#" -ge 2 ] || { echo "error: --alpamayo requires a path" >&2; exit 2; }
            ALPAMAYO_VIDEO="$2"
            shift 2
            ;;
        --visionpilot)
            [ "$#" -ge 2 ] || { echo "error: --visionpilot requires a path" >&2; exit 2; }
            VISIONPILOT_VIDEO="$2"
            shift 2
            ;;
        --output)
            [ "$#" -ge 2 ] || { echo "error: --output requires a path" >&2; exit 2; }
            OUTPUT_VIDEO="$2"
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

for command in ffmpeg ffprobe; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "error: required command is missing: $command" >&2
        exit 1
    }
done

for input in "$ALPAMAYO_VIDEO" "$VISIONPILOT_VIDEO"; do
    [ -s "$input" ] || { echo "error: input video is missing or empty: $input" >&2; exit 1; }
done

filter_list="$(ffmpeg -hide_banner -filters 2>&1)"
if [[ "$filter_list" != *drawtext* ]]; then
    echo "error: FFmpeg drawtext support is required" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_VIDEO")"

filter="[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
filter+="pad=960:540:(ow-iw)/2:(oh-ih)/2,setsar=1,"
filter+="drawtext=text='Alpamayo 1.5 VLA - multi-camera':x=24:y=22:fontsize=28:"
filter+="fontcolor=white:box=1:boxcolor=black@0.72:boxborderw=10[a];"
filter+="[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
filter+="pad=960:540:(ow-iw)/2:(oh-ih)/2,setsar=1,"
filter+="drawtext=text='VisionPilot - single-camera ONNX':x=24:y=22:fontsize=28:"
filter+="fontcolor=white:box=1:boxcolor=black@0.72:boxborderw=10[b];"
filter+="[a][b]hstack=inputs=2:shortest=1[out]"

ffmpeg -y \
    -i "$ALPAMAYO_VIDEO" \
    -i "$VISIONPILOT_VIDEO" \
    -filter_complex "$filter" \
    -map "[out]" \
    -an \
    -r 5 \
    -c:v libx264 \
    -crf 20 \
    -pix_fmt yuv420p \
    "$OUTPUT_VIDEO"

ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
    -of json \
    "$OUTPUT_VIDEO"

echo "Unified showcase written to: $OUTPUT_VIDEO"
