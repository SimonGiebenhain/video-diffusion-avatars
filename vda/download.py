"""Downloads: our weights, the public WAN base files, the ready-made examples and the example driving videos.

    python download.py weights                        # our weights into <weights>, md5-checked, ~6.5 GB
    python download.py base [--source huggingface]    # WAN files into <weights>/base (DiffSynth layout), ~37 GB
    python download.py examples [--names <NAME> ...]  # examples + driving videos into <checkout>/examples, ~220 MB

Base files: Wan2.2-Animate-14B DiT shards (+ config, and its bundled stock animate adapter), Wan2.1 VAE + open-clip
ViT-H image encoder (DiffSynth's converted safetensors), the umt5 tokenizer. The umt5-xxl TEXT ENCODER (11 GB) is
NOT needed: weights/text_emb_cache.pt holds the prompt embeddings (--with_text_encoder fetches it for new prompts).

Examples: each examples/<name>/ holds what python generate.py needs -- ref.png (720x1280 reference), pose_proxy.mp4 (proxy
render), pose_skel.mp4 (skeleton), codes.npy (X-NeMo expression codes) -- plus meta.json (subject, driver, camera,
model). No preprocessing or nvdiffrast is needed to run them.
  teaser_*          7 examples, full model (77 frames)
  control_camera_*  3 examples, full model: camera control, static photo expression
  control_neck_*    4 examples, full model: neck pose control, static camera
  control_expr_*    3 examples, full model: expression control while the camera moves
  distilled_*       8 examples, fast model: the 41 s driving sequence nersemble8 (1029 frames)
Driving videos: examples/drivers/<name>.mp4, the NeRSemble performances behind the examples, as inputs for
prepare_driving.py (always downloaded with the examples).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

from . import paths

HF_REPO = "SimonGiebenhain/video-diffusion-avatars"   # Hugging Face model repo: weights, examples/, third_party/

#: our weights in the Hugging Face repo, paths relative to <weights> (README "Weights")
WEIGHT_FILES = ["MD5SUMS.txt", "vda_full.safetensors", "vda_fast_base.safetensors", "vda_fast_student.safetensors",
                "xnemo_disentangler.pt", "text_emb_cache.pt", "driving_bank/nersemble8.npz",
                "driving_bank/nersemble8.takes.json", "driving_bank/nersemble8_canonical_body.npz",
                "third_party/xnemo_motion_encoder.pth"]

EXAMPLES = ["teaser_swimmer", "teaser_gele", "teaser_greenhair", "teaser_hijab", "teaser_cartoon",
            "teaser_grandma", "teaser_vangogh",
            "control_camera_tortoise", "control_camera_curlyboy", "control_camera_refraction",
            "control_neck_neon_earring", "control_neck_red_hair_teen", "control_neck_straw_hat",
            "control_neck_beanie_locs",
            "control_expr_swimmer", "control_expr_blinds", "control_expr_gele",
            "distilled_vangogh", "distilled_marble", "distilled_watchmaker", "distilled_frida",
            "distilled_cgrandma", "distilled_refraction", "distilled_redbangs",
            "distilled_greenhair"]


def check_md5(folder):
    """Compare every file listed in <folder>/MD5SUMS.txt; returns the names that are missing or differ."""
    bad = []
    for line in (Path(folder) / "MD5SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        md5, name = line.split(maxsplit=1)
        p = Path(folder) / name
        h = hashlib.md5()
        if p.is_file():
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
        if not p.is_file() or h.hexdigest() != md5:
            bad.append(name)
        print(f"[weights] {name}: {'OK' if name not in bad else 'MISSING OR WRONG'}", flush=True)
    return bad


def weights_main(argv=None):
    ap = argparse.ArgumentParser(prog="download.py weights",
                                 description="Download our weights (~6.5 GB) from Hugging Face and check their md5.")
    ap.add_argument("--no_check", action="store_true", help="skip the md5 check")
    a = ap.parse_args(argv)
    if HF_REPO.startswith("<"):
        sys.exit("download.py weights: HF_REPO is not set yet (placeholder in vda/download.py).")
    from huggingface_hub import snapshot_download
    dst = paths.weights_dir()
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, allow_patterns=WEIGHT_FILES, local_dir=str(dst))
    if not a.no_check and check_md5(dst):
        sys.exit("download.py weights: some files are missing or corrupt; run the command again")
    print(f"[weights] done -> {dst}")


def base_main(argv=None):
    ap = argparse.ArgumentParser(prog="download.py base", description="Download the public WAN files (~37 GB).")
    ap.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    ap.add_argument("--with_text_encoder", action="store_true", help="also fetch umt5-xxl (only for new prompts)")
    a = ap.parse_args(argv)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(paths.weights_dir() / "base")
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = a.source
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "false"
    from .third_party.diffsynth.core import ModelConfig
    A = "Wan-AI/Wan2.2-Animate-14B"
    C = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
    cfgs = [ModelConfig(model_id=A, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=A, origin_file_pattern="*.json"),
            ModelConfig(model_id=C, origin_file_pattern="Wan2.1_VAE.safetensors"),
            ModelConfig(model_id=C, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")]
    if a.with_text_encoder:
        cfgs.append(ModelConfig(model_id=C, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors"))
    for c in cfgs:
        print(f"[download] {c.model_id} / {c.origin_file_pattern}", flush=True)
        c.download_if_necessary()
    print(f"[download] done -> {os.environ['DIFFSYNTH_MODEL_BASE_PATH']}")


#: example driving videos (examples/drivers/<name>.mp4): NeRSemble performances, 720x1280, 81 frames at 24 fps
DRIVERS = ["disgust_happy", "tongue_out", "shout_laugh", "jaw", "surprise_fear", "cheeks_nose"]


def examples_main(argv=None):
    ap = argparse.ArgumentParser(prog="download.py examples", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", choices=EXAMPLES, help="download only these examples (default: all)")
    ap.add_argument("--dir", help="target examples folder (default: <checkout>/examples, else ./examples)")
    ap.add_argument("--list", action="store_true", help="print the example names and exit")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(EXAMPLES))
        return
    if HF_REPO.startswith("<"):
        sys.exit("download.py examples: HF_REPO is not set yet (placeholder in vda/download.py).")
    from huggingface_hub import snapshot_download
    dst = Path(a.dir).expanduser().resolve() if a.dir else paths.examples_dir()
    names = a.names or EXAMPLES
    # the repo stores them under examples/<name>/: download next to dst, then the folder is dst itself
    stage = dst.parent if dst.name == "examples" else dst / "_hf"
    snapshot_download(repo_id=HF_REPO, allow_patterns=[f"examples/{n}/*" for n in names] + ["examples/drivers/*"],
                      local_dir=str(stage))
    if stage != dst.parent:
        dst.mkdir(parents=True, exist_ok=True)
        for n in names + ["drivers"]:
            (stage / "examples" / n).replace(dst / n)
    missing = [n for n in names if not (dst / n / "codes.npy").exists()]
    missing += [f"drivers/{d}.mp4" for d in DRIVERS if not (dst / "drivers" / f"{d}.mp4").exists()]
    if missing:
        sys.exit(f"download incomplete, missing: {missing}")
    print(f"[examples] {len(names)} example(s) and {len(DRIVERS)} driving videos in {dst}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("weights", "base", "examples"):
        sys.exit("usage: python download.py weights|base|examples [options]")
    {"weights": weights_main, "base": base_main, "examples": examples_main}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
