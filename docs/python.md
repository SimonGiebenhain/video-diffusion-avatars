# Scripts, paths and the Python package

Every step is a Python script in the repository folder that calls the code in the package `vda/`; every script has
`--help`. `envs/setup_env.sh` also installs the repository as the editable package `video-diffusion-avatars`
(`pip install -e .`), so `import vda` works from other repositories too.

## Scripts

| script | code | what it does |
|---|---|---|
| `python download.py weights` | `vda/download.py` | our weights into `weights/` (~6.5 GB), md5-checked |
| `python download.py base` | `vda/download.py` | public WAN2.2-Animate-14B files into `weights/base/` (~37 GB) |
| `python download.py examples` | `vda/download.py` | the [examples](examples.md) and the example driving videos into `examples/` (~220 MB) |
| `python prepare_reference.py` | `vda/preprocess/reference.py` | crop a photo and fit its FLAME shape ([reference_images.md](reference_images.md)) |
| `python prepare_driving.py` | `vda/preprocess/driving.py` | prepare a driving video ([driving.md](driving.md)) |
| `python render_conditioning.py` | `vda/preprocess/conditioning.py` | proxy render, skeleton and codes for (reference, driving, camera) ([cameras.md](cameras.md)) |
| `python generate.py` | `vda/generate.py` | the video, with the full or the fast model |
| `bash scripts/run_example.sh` | | all steps for a new photo |
| `python scripts/prepare_body_motion.py` | `vda/preprocess/body_motion.py` | SAM 3D Body poses for `prepare_driving.py --motion driver`; run it in the SAM 3D Body env, from the repository folder |
| `python scripts/extract_xnemo_codes.py` | `vda/preprocess/xnemo_codes.py` | X-NeMo expression codes [T, 512] of a video |
| `python scripts/disentangle.py` | `vda/disentangler.py` | split codes into expression and view parts (below) |
| `python scripts/deflicker.py` | `vda/postprocess/deflicker.py` | the block-seam colour fix for fast-model videos (also `generate.py --deflicker`) |
| `python scripts/add_prompt.py` | `vda/text_cache.py` | embeddings for a new prompt (untested; the models were trained with one fixed prompt) |
| `python scripts/plot_cameras.py` | `vda/preprocess/camera.py` | the camera preset figure ([cameras.md](cameras.md)) |

## Where files go

| what | default | override |
|---|---|---|
| weights | `<checkout>/weights/` | `VDA_WEIGHTS=<DIR>` |
| preprocessing cache (references, driving videos, conditionings) | `<checkout>/cache/` | `VDA_CACHE=<DIR>` |
| examples | `<checkout>/examples/` | `python download.py examples --dir <DIR>` |
| pixel3dmm (preprocessing) | `<checkout>/third_party/pixel3dmm/` | `PIXEL3DMM_CODE_BASE=<DIR>` |
| torch hub / Hugging Face model caches (~1.2 GB, preprocessing) | `~/.cache/` | `TORCH_HOME`, `HF_HOME` |

Names on the command line are looked up in these folders: `--ref <NAME>` in `cache/refs/`, `--driving <DRV>` in
`cache/driving/`, `--driving nersemble8` in `weights/driving_bank/`. Paths work everywhere too.

## Using VDA from another repository

Install the checkout into your environment; `--no-deps` keeps your packages as they are:

```bash
pip install --no-deps -e <path/to/video-diffusion-avatars>
```

Generation needs the versions pinned in `envs/requirements_generate.txt` (torch 2.10.0 + CUDA 12.8, transformers,
peft, ...); `pip install -e "<checkout>[generate]"` installs them (torch first, from the CUDA 12.8 index). Other
versions may work, but the outputs then no longer match ours bit for bit. Preprocessing needs the full env from
`envs/setup_env.sh`.

```python
from vda import paths
from vda.model import load_pipeline
from vda.generate import default_offload_blocks, full_num_frames, generate_full

cond = paths.examples_dir() / "teaser_swimmer"        # any conditioning folder; examples_dir() = <checkout>/examples
pipe = load_pipeline(paths.weight(paths.FULL_ADAPTER))   # weights from $VDA_WEIGHTS or <checkout>/weights
generate_full(pipe, str(cond / "ref.png"), str(cond / "pose_skel.mp4"), str(cond / "pose_proxy.mp4"),
              str(cond / "codes.npy"), "out.mp4", num_frames=full_num_frames(cond / "pose_skel.mp4"),
              offload_blocks=default_offload_blocks("full", 1024, 80))   # 0 on 80 GB GPUs
```

`vda/generate.py` (`main()`) shows the fast model (`generate_fast`, `load_student_lora`).

## Expression disentangler

X-NeMo expression codes depend on the viewpoint: the same expression filmed from two cameras gives different codes.
The disentangler, a small MLP trained on NeRSemble's 16 synchronised cameras, splits a code into a view-invariant
expression part and a view part, and recombines them. Inside the video model only its expression encoder is used.

```python
import numpy as np
from vda import paths
from vda.disentangler import XNemoDisentangler

ex = paths.examples_dir()
dis = XNemoDisentangler.load(paths.weight(paths.DISENTANGLER))
codes = np.load(ex / "teaser_swimmer" / "codes.npy")              # [T, 512] raw X-NeMo codes
z_expr, z_cam = dis.disentangle(codes)                             # [T, 512] expression, [T, 32] view
_, z_cam_b = dis.disentangle(np.load(ex / "control_camera_tortoise" / "codes.npy"))   # the views of another clip
codes_b = dis.entangle(z_expr, z_cam_b[: len(z_expr)])             # this expression, seen from those views
```

On the command line: `python scripts/disentangle.py <CODES.npy> --out <Z.npz>` (writes `z_expr`, `z_cam` and the reconstruction);
codes of any video: `python scripts/extract_xnemo_codes.py --video <VIDEO.mp4> --out <CODES.npy>`.
