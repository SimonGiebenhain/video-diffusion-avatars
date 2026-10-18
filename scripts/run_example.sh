#!/bin/bash
# All steps for a new photo: prepare the reference and a driving video, render the conditioning with the
# over_the_top_zoom camera and generate it with the full model.
#   bash scripts/run_example.sh <PHOTO.jpg> <NAME> [--driver <DRIVER.mp4>] [--min_conf 0.25] [--fast]
# <NAME> is any short name for the person (cache folder and output name). The default driving video is
# examples/drivers/disgust_happy.mp4 (python download.py examples). --fast additionally renders the shipped 41 s
# driving sequence nersemble8 with the fast model (README section 3).
set -e
usage="usage: run_example.sh <PHOTO.jpg> <NAME> [--driver <DRIVER.mp4>] [--min_conf 0.25] [--fast]"
IMG=${1:?$usage}
NAME=${2:?$usage}
shift 2
FAST=0; DRIVER=""; REF_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1 ;;
        --driver) DRIVER=$(realpath "$2"); shift ;;
        *) REF_ARGS+=("$1") ;;
    esac
    shift
done
IMG=$(realpath "$IMG")
cd "$(dirname "$0")/.."
DRIVER=${DRIVER:-examples/drivers/disgust_happy.mp4}
[ -f "$DRIVER" ] || { echo "driving video $DRIVER not found (python download.py examples)"; exit 1; }
DRV=$(basename "${DRIVER%.*}")
CACHE=${VDA_CACHE:-cache}
PY=python                                     # the active environment (conda activate vda, README 1.2)
$PY -c "import torch, nvdiffrast.torch" 2>/dev/null || { echo "activate the VDA environment first (conda activate vda)"; exit 1; }
$PY prepare_reference.py --image "$IMG" --name "$NAME" "${REF_ARGS[@]}"
$PY prepare_driving.py --video "$DRIVER" --name "$DRV"
$PY render_conditioning.py --ref "$NAME" --driving "$DRV" --camera over_the_top_zoom \
    --out_dir "$CACHE/cond/${NAME}__${DRV}__over_the_top_zoom"
$PY generate.py --cond "$CACHE/cond/${NAME}__${DRV}__over_the_top_zoom" --out "outputs/${NAME}.mp4"
echo "outputs/${NAME}.mp4"
if [ "$FAST" = 1 ]; then
    $PY render_conditioning.py --ref "$NAME" --driving nersemble8 --camera sweep_41s \
        --out_dir "$CACHE/cond/${NAME}__nersemble8__sweep_41s"
    $PY generate.py --model fast --cond "$CACHE/cond/${NAME}__nersemble8__sweep_41s" \
        --out "outputs/${NAME}_41s_fast.mp4" --deflicker
    echo "outputs/${NAME}_41s_fast_dfs.mp4"
fi
