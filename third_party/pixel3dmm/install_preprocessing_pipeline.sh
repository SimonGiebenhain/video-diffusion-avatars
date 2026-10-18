#!/bin/bash
# Installs what pixel3dmm's preprocessing and tracking need, as VDA uses them: facer (face segmentation),
# PIPNet (face detection + landmarks), FLAME2020 (+ the FLAME masks / landmark embedding from the MICA repo)
# and the pixel3dmm uv / normal networks. Third-party code goes to src/pixel3dmm/preprocessing/, the networks
# to pretrained_weights/. Safe to re-run: finished steps are skipped and existing checkouts are reused.
#
#   FLAME_USERNAME, FLAME_PASSWORD   login for https://flame.is.tue.mpg.de (prompted if unset)
#   FLAME_ZIP_DIR                    a directory with a pre-downloaded FLAME2020.zip (or its extracted
#                                    generic_model.pkl): no download, no login needed
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$HERE/src/pixel3dmm/preprocessing
RC=$PRE/replacement_code

clone() {   # clone <https url> <dir> <commit>: clone once, then pin the commit
    local url=$1 dir=$2 commit=$3
    [ -d "$dir/.git" ] || git clone "$url" "$dir"
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
        git -C "$dir" checkout --force --quiet "$commit"
    fi
}

md5of() { md5sum "$1" | cut -d' ' -f1; }

fetch() {   # fetch <output> <md5> <source>...: first source whose file has the right md5 wins
    # a source is an https URL or gdrive:<Google Drive file id>; an existing file with the right md5 is kept
    local out=$1 md5=$2 src; shift 2
    [ -s "$out" ] && [ "$(md5of "$out")" = "$md5" ] && return 0
    mkdir -p "$(dirname "$out")"
    for src in "$@"; do
        rm -f "$out.part"
        echo "downloading $(basename "$out") from $src"
        case "$src" in
            gdrive:*) gdown "${src#gdrive:}" -O "$out.part" || true ;;
            *) if command -v curl >/dev/null; then curl -fL --retry 3 -o "$out.part" "$src" || true
               else wget -O "$out.part" "$src" || true; fi ;;
        esac
        if [ -s "$out.part" ] && [ "$(md5of "$out.part")" = "$md5" ]; then mv "$out.part" "$out"; return 0; fi
        echo "  -> failed or wrong checksum"
    done
    rm -f "$out.part"
    echo "ERROR: could not download $(basename "$out") (md5 $md5) from any of: $*" >&2
    echo "  Google Drive limits downloads per file and day ('Too many users have viewed or downloaded this file')." >&2
    echo "  Retry later, or download it in a browser, save it as $out and re-run this script." >&2
    exit 1
}

# ------------------------------------------------------------------ facer (face parsing)
clone https://github.com/FacePerceiver/facer.git "$PRE/facer" ddd35c76ff840174b8a5403ad1c1255e37b8782b
cp "$RC/farl.py" "$PRE/facer/facer/face_parsing/farl.py"
cp "$RC/facer_transform.py" "$PRE/facer/facer/transform.py"
python -c "import facer" 2>/dev/null || pip install -e "$PRE/facer"

# ------------------------------------------------------------------ PIPNet (face detection + landmarks)
clone https://github.com/jhb86253817/PIPNet.git "$PRE/PIPNet" b9eab58816437403a34aa5bc3adeafe5081fd36b
if ! ls "$PRE"/PIPNet/FaceBoxesV2/utils/nms/cpu_nms*.so >/dev/null 2>&1; then
    (cd "$PRE/PIPNet/FaceBoxesV2/utils" && sh make.sh)
fi
fetch "$PRE/PIPNet/snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth" af44570e01de8bd19ff63389ac67a965 \
    https://huggingface.co/SimonGiebenhain/video-diffusion-avatars/resolve/main/third_party/pipnet/epoch59.pth \
    gdrive:1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C

# ------------------------------------------------------------------ FLAME2020
# The MICA repo provides the FLAME masks and landmark embedding next to the model (only its data/ folder is
# used; MICA itself is not run).
clone https://github.com/Zielon/MICA.git "$PRE/MICA" af22e7a5810d474bc28a1433db533723d6bd2b07
FLAME_DIR=$PRE/MICA/data/FLAME2020
flame_notice() {
    cat >&2 <<EOF

  FLAME is required. pixel3dmm fits the FLAME 2020 head model, which is licensed by MPI-IS and cannot be
  redistributed: register for free at https://flame.is.tue.mpg.de and accept the licence. This step then
  downloads FLAME2020.zip and unpacks it to
      $FLAME_DIR
  Instead of typing the login you can set FLAME_USERNAME and FLAME_PASSWORD, or set FLAME_ZIP_DIR to a
  folder that already contains FLAME2020.zip.

EOF
}
flame_failed() { echo "ERROR: FLAME download failed (wrong credentials or licence not accepted)." >&2; flame_notice; exit 1; }
if [ ! -s "$FLAME_DIR/generic_model.pkl" ]; then
    if [ -n "$FLAME_ZIP_DIR" ] && [ -s "$FLAME_ZIP_DIR/generic_model.pkl" ]; then
        cp "$FLAME_ZIP_DIR/generic_model.pkl" "$FLAME_DIR/"
    else
        ZIP=$PRE/MICA/FLAME2020.zip
        if [ -n "$FLAME_ZIP_DIR" ] && [ -s "$FLAME_ZIP_DIR/FLAME2020.zip" ]; then
            ZIP=$FLAME_ZIP_DIR/FLAME2020.zip
        else
            if [ -z "$FLAME_USERNAME" ] || [ -z "$FLAME_PASSWORD" ]; then
                flame_notice
                if [ ! -t 0 ]; then
                    echo "ERROR: no terminal to ask for the FLAME login; set FLAME_USERNAME / FLAME_PASSWORD or FLAME_ZIP_DIR." >&2
                    exit 1
                fi
                [ -n "$FLAME_USERNAME" ] || read -r -p "FLAME username (e-mail): " FLAME_USERNAME
                [ -n "$FLAME_PASSWORD" ] || { read -r -s -p "FLAME password: " FLAME_PASSWORD; echo; }
            fi
            urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
            rm -f "$ZIP"
            wget -q --post-data "username=$(urle "$FLAME_USERNAME")&password=$(urle "$FLAME_PASSWORD")" \
                'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
                -O "$ZIP" --no-check-certificate || { rm -f "$ZIP"; flame_failed; }
        fi
        unzip -tq "$ZIP" >/dev/null 2>&1 || { [ "$ZIP" = "$PRE/MICA/FLAME2020.zip" ] && rm -f "$ZIP"; flame_failed; }
        unzip -oq "$ZIP" -d "$FLAME_DIR"
        [ "$ZIP" = "$PRE/MICA/FLAME2020.zip" ] && rm -f "$ZIP"
        [ -s "$FLAME_DIR/generic_model.pkl" ] || flame_failed
    fi
fi

# ------------------------------------------------------------------ pixel3dmm uv / normal networks (CC BY-NC 4.0)
fetch "$HERE/pretrained_weights/uv.ckpt" 8b6dfa2e9e42dbd7ee338fe0e2cbd856 \
    https://huggingface.co/SimonGiebenhain/Pixel3dmm/resolve/main/uv.ckpt gdrive:1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY
fetch "$HERE/pretrained_weights/normals.ckpt" 86e1dad3392a290d91fa3bbdc2ce03ac \
    https://huggingface.co/SimonGiebenhain/Pixel3dmm/resolve/main/normals.ckpt gdrive:1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD

for f in "$FLAME_DIR/generic_model.pkl" "$FLAME_DIR/landmark_embedding.npy" "$FLAME_DIR/FLAME_masks/FLAME_masks.pkl"; do
    [ -s "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
echo "pixel3dmm preprocessing installed."
