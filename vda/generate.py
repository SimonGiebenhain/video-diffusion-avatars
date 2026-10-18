"""Generate a video from (reference image, conditioning directory).

A conditioning directory (written by `vda.preprocess.conditioning`) holds three frame-aligned streams:
    pose_skel.mp4   2D AAPose skeleton in the target camera
    pose_proxy.mp4  proxy render: FLAME head (reference shape, driver expression) on the body in the target camera
    codes.npy       RAW X-NeMo expression codes of the driving frames [T, 512]

Models
  full  the VDA adapter, bidirectional, 77 output frames per call: 20 ODE steps with JOINT guidance
        (identity + expression together, scale 1.5, 2 DiT passes per step). Longer driving sequences: use
        the fast model.
  fast  the distilled model: causal 4-step Self-Forcing student, streams any length on a KV cache
        (reference sink + rolling window), first 77 frames by its bidirectional teacher. ~5 fps at 288x512
        on an A100-80GB.

Example
  python generate.py --model fast --cond cache/cond/me__41s__sweep --out out.mp4 --deflicker
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageFilter

from . import paths
from .model import DEFAULT_PROMPT, enable_block_offload, load_pipeline, load_student_lora

#: the full model's sampling: 20 Euler (ODE) steps, joint identity+expression guidance at 1.5
#: (eps = base + 1.5 * (cond - base), base = blurred reference with the expression dropped)
FULL = dict(num_inference_steps=20, cfg_joint=True, cfg_scale=1.0, cfg_scale_id=1.5, cfg_scale_graft=1.0)
ID_BLUR_SIGMA = 8.0          # weak-identity reference for the guidance base pass (as trained)
OOT_FILLER = 128             # generate-region fill of the inpaint video (mid-gray, as trained)


# ----------------------------------------------------------------------------------------- io
def load_frames(path, size, n=None):
    from vda.third_party.diffsynth.utils.data import VideoData
    frames = VideoData(path).raw_data()
    if n is not None:
        frames = frames[:n]
    return [Image.fromarray(np.asarray(f)).convert("RGB").resize(size) for f in frames]


def load_ref(path, size):
    from vda.third_party.diffsynth.utils.data import VideoData
    if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        return Image.open(path).convert("RGB").resize(size)
    return Image.fromarray(np.asarray(VideoData(path).raw_data()[0])).convert("RGB").resize(size)


def save_outputs(frames, out, fps, png):
    from vda.third_party.diffsynth.utils.data import save_video
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_video(list(frames), out, fps=fps, quality=6)
    if png:
        d = os.path.splitext(out)[0]
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(np.asarray(f)).convert("RGB").save(os.path.join(d, f"{i:05d}.png"))
    print(f"[vda] saved {len(frames)} frames -> {out}{' (+ png frames)' if png else ''}", flush=True)


def decode_long(pipe, lat, seg=20, warm=3, tiled=False):
    """Decode a long latent sequence with the causal VAE in segments of `seg` slots; every segment
    after the first re-warms the decoder on `warm` earlier slots, whose frames are dropped."""
    outs, a, n = [], 0, int(lat.shape[2])
    while a < n:
        b = min(a + seg, n)
        w0 = max(0, a - warm)
        vid = pipe.vae.decode(lat[:, :, w0:b], device=pipe.device, tiled=tiled)
        drop = 0 if a == 0 else 1 + 4 * (a - w0 - 1)
        outs.append(vid[:, :, drop:].float().cpu())
        a = b
    return torch.cat(outs, dim=2)


def _oot(size, n):
    return dict(animate_inpaint_video=[Image.new("RGB", size, (OOT_FILLER,) * 3)] * n,
                animate_mask_video=[Image.new("RGB", size, (255, 255, 255))] * n)


# ------------------------------------------------------------------------------- full model
def full_num_frames(pose_path, max_out=77):
    """WAN frame count for one full-model window: 81 (= 77 output frames), or fewer when the conditioning is
    shorter (output frames must be 4k+1)."""
    import cv2
    if not os.path.isfile(pose_path):
        raise SystemExit(f"conditioning file not found: {pose_path}")
    cap = cv2.VideoCapture(str(pose_path))
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_out)
    cap.release()
    if n < 5:
        raise SystemExit(f"conditioning too short: {n} frames in {pose_path}")
    return n - (n - 1) % 4 + 4


def generate_full(pipe, ref_path, pose_path, proxy_path, codes_path, out, width=576, height=1024,
                  num_frames=81, seed=42, tiled=True, fps=24, png=False, offload_blocks=0):
    """One window: num_frames-4 output frames (77 for num_frames=81)."""
    enable_block_offload(pipe, offload_blocks)      # weights are final here (adapter loaded, no LoRA)
    size = (width, height)
    r = FULL
    n_cond = num_frames - 4
    pose = load_frames(pose_path, size, n_cond)
    proxy = load_frames(proxy_path, size, n_cond)
    if len(proxy) != len(pose):
        raise SystemExit(f"proxy/pose length mismatch: {len(proxy)} vs {len(pose)}")
    codes = torch.from_numpy(np.load(codes_path)).float()
    ref = load_ref(ref_path, size)
    ref_weak = ref.filter(ImageFilter.GaussianBlur(ID_BLUR_SIGMA)) if r["cfg_scale_id"] != 1.0 else None
    prompt, negative = pipe.vda_prompts
    video = pipe(prompt=prompt, negative_prompt=negative, input_image=ref,
                 animate_pose_video=pose, animate_graft_video=proxy, expr_codes=codes,
                 height=height, width=width, num_frames=num_frames, id_weak_image=ref_weak,
                 sampler="ode", seed=seed, tiled=tiled,
                 **_oot(size, n_cond), **r)
    save_outputs(video, out, fps, png)
    return video


# ------------------------------------------------------------------------------- fast model
FAST = dict(num_inference_steps=4, kv_window=20, sink_slots=12, ref_boost=4, block=4,
            teacher_prefix_slots=20, teacher_steps=10, teacher_cfg=(2.0, 1.5, 1.6))


def generate_fast(pipe, student_path, ref_path, pose_path, proxy_path, codes_path, out, width=288, height=512,
                  seed=42, tiled=True, fps=24, png=False, offload_blocks=0, max_frames=0,
                  teacher_prefix=True, cfg=FAST):
    """Stream the whole driving sequence in ONE latent timeline on the KV cache."""
    size = (width, height)
    prompt, negative = pipe.vda_prompts
    ref = load_ref(ref_path, size)

    # 1) teacher-quality anchor: the first 77 frames by the bidirectional teacher (the base adapter,
    #    before the student LoRA is merged); they become the never-evicted sink of the cache.
    prefill = None
    if teacher_prefix:
        t0 = time.time()
        P = int(cfg["teacher_prefix_slots"])
        nf_t = min(81, 4 * (P - 1) + 1 + 4)
        nc = nf_t - 4
        ce, ci_, cg = cfg["teacher_cfg"]
        lat = pipe(prompt=prompt, negative_prompt=negative, input_image=ref,
                   animate_pose_video=load_frames(pose_path, size, nc),
                   animate_graft_video=load_frames(proxy_path, size, nc),
                   expr_codes=torch.from_numpy(np.load(codes_path).astype(np.float32)[:nc]).float(),
                   height=height, width=width, num_frames=nf_t, num_inference_steps=int(cfg["teacher_steps"]),
                   cfg_scale=ce, cfg_scale_id=ci_, id_weak_image=ref.filter(ImageFilter.GaussianBlur(ID_BLUR_SIGMA)),
                   cfg_scale_graft=cg, sampler="ode", seed=seed, tiled=tiled,
                   return_latents=True, **_oot(size, nc))
        prefill = lat[:, :, :min(P, int(lat.shape[2]))].detach()
        del lat
        torch.cuda.empty_cache()
        print(f"[vda] teacher prefix: {prefill.shape[2]} latent slots in {time.time() - t0:.1f}s", flush=True)

    # 2) the student (irreversible merge), then optional weight streaming for high resolutions
    load_student_lora(pipe, student_path)
    enable_block_offload(pipe, offload_blocks)

    # 3) one pipeline call over the whole sequence (4k+1 frames -> continuous latent timeline)
    pose_all = load_frames(pose_path, size)
    if max_frames > 0:
        pose_all = pose_all[:max_frames]
    n_total = len(pose_all)
    proxy_all = load_frames(proxy_path, size)
    if len(proxy_all) < n_total:
        raise SystemExit(f"proxy stream too short: {len(proxy_all)} < {n_total}")
    codes_np = np.load(codes_path).astype(np.float32)
    assert codes_np.shape[0] >= n_total, f"codes stream too short: {codes_np.shape[0]} < {n_total}"
    n_cond = ((n_total - 1 + 3) // 4) * 4 + 1

    def _pad(seq):
        seq = seq[:n_cond]
        return seq + [seq[-1]] * (n_cond - len(seq))
    cc = codes_np[:n_cond]
    if len(cc) < n_cond:
        cc = np.concatenate([cc, np.repeat(cc[-1:], n_cond - len(cc), axis=0)], axis=0)
    state = {"chunk": int(cfg["block"]), "window": int(cfg["kv_window"]), "caches": None,
             "sink": int(cfg["sink_slots"]), "ref_dup": int(cfg["ref_boost"]), "prefill": prefill}
    t0 = time.time()
    lat = pipe(prompt=prompt, negative_prompt=negative, input_image=ref,
               animate_pose_video=_pad(pose_all), animate_graft_video=_pad(proxy_all),
               expr_codes=torch.from_numpy(cc).float(), height=height, width=width, num_frames=n_cond + 4,
               num_inference_steps=int(cfg["num_inference_steps"]), cfg_scale=1.0, cfg_scale_id=1.0,
               cfg_scale_graft=1.0, sampler="sde", seed=seed, tiled=tiled, kv_stream=state,
               **_oot(size, n_cond))
    print(f"[vda] streamed {n_total} frames in {time.time() - t0:.0f}s; peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    state["caches"] = None
    torch.cuda.empty_cache()
    n_slots = (n_total - 1 + 3) // 4 + 1
    video = decode_long(pipe, lat.cpu()[:, :, :n_slots].to(pipe.device), seg=state["window"], warm=3, tiled=tiled)
    frames = ((video[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).numpy().astype(np.uint8)[:n_total]
    save_outputs(frames, out, fps, png)
    return frames


# --------------------------------------------------------------------------------------- CLI
def default_offload_blocks(model, height, gib):
    """DiT blocks streamed from host memory (enable_block_offload) when --offload_blocks is not given.
    full: everything resident fits a 48 GB GPU up to 576x1024 (43 GB peak); 720x1280 on a 48 GB GPU streams 20
    blocks (40 GB peak, measured on an RTX A6000).
    fast: the KV cache grows with resolution (~11 GB at 288p, ~45 GB at 576p, ~70 GB at 720p)."""
    if model == "full":
        return 0 if gib >= 60 or height <= 1024 else 20
    n = 0 if height <= 512 else (20 if height <= 1024 else 36)
    return max(n, 12) if gib < 60 else n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["full", "fast"], default="full")
    ap.add_argument("--cond", help="conditioning dir with pose_skel.mp4, pose_proxy.mp4, codes.npy (+ ref.png)")
    ap.add_argument("--ref", help="reference: a 720x1280 portrait crop (see prepare_reference.py), its folder or its "
                                  "prepare_reference.py name; default <cond>/ref.png")
    ap.add_argument("--pose"); ap.add_argument("--proxy"); ap.add_argument("--codes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--png", action="store_true", help="also write lossless PNG frames next to --out")
    ap.add_argument("--vae_tiled", choices=["on", "off"], default="on",
                    help="tiled VAE (on = reproducible across GPU types; off = faster on 80 GB)")
    ap.add_argument("--offload_blocks", type=int, default=None,
                    help="DiT blocks (of 40) kept in host memory and streamed to the GPU for each forward: "
                         "lower peak VRAM, same output, slower. Default: full 0 (20 at 720x1280 on <60 GB GPUs); "
                         "fast 0 at 288p, 20 at 576p, 36 at 720p, at least 12 on <60 GB GPUs")
    ap.add_argument("--max_frames", type=int, default=0, help="fast: stream only the first N frames")
    ap.add_argument("--no_teacher_prefix", action="store_true", help="fast: skip the bidirectional teacher anchor")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--deflicker", action="store_true", help="fast: also write <out>_dfs.mp4 (block-seam colour fix)")
    a = ap.parse_args()
    if not a.ref and not a.cond:
        raise SystemExit("pass --cond <dir> (with ref.png) or --ref")
    a.ref = paths.reference_image(a.ref or a.cond)
    if a.cond:
        a.pose = a.pose or os.path.join(a.cond, "pose_skel.mp4")
        a.proxy = a.proxy or os.path.join(a.cond, "pose_proxy.mp4")
        a.codes = a.codes or os.path.join(a.cond, "codes.npy")
    for k in ("pose", "proxy", "codes"):
        if not getattr(a, k) or not os.path.exists(getattr(a, k)):
            raise SystemExit(f"missing --{k} (or --cond dir) : {getattr(a, k)}")
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    tiled = a.vae_tiled == "on"
    if a.model == "full":
        w, h = a.width or 576, a.height or 1024
    else:
        w, h = a.width or 288, a.height or 512
    a.width, a.height = w, h                      # recorded in <out>.json as used
    if a.offload_blocks is None:
        a.offload_blocks = default_offload_blocks(a.model, h, gib)
    off = a.offload_blocks
    if a.model == "full":
        pipe = load_pipeline(paths.weight(paths.FULL_ADAPTER), prompt=a.prompt)
        generate_full(pipe, a.ref, a.pose, a.proxy, a.codes, a.out, width=w, height=h, seed=a.seed,
                      num_frames=full_num_frames(a.pose), tiled=tiled, fps=a.fps, png=a.png, offload_blocks=off)
    else:
        pipe = load_pipeline(paths.weight(paths.FAST_BASE_ADAPTER), prompt=a.prompt)
        generate_fast(pipe, paths.weight(paths.FAST_STUDENT), a.ref, a.pose, a.proxy, a.codes, a.out,
                      width=w, height=h, seed=a.seed, tiled=tiled, fps=a.fps, png=a.png, offload_blocks=off,
                      max_frames=a.max_frames, teacher_prefix=not a.no_teacher_prefix)
        if a.deflicker:
            from .postprocess.deflicker import deflicker
            deflicker(a.out, os.path.splitext(a.out)[0] + "_dfs.mp4")
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump({k: v for k, v in vars(a).items()}, f, indent=1)


if __name__ == "__main__":
    main()
