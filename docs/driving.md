# Driving videos

The driving video says what the person does: the facial expression (as X-NeMo expression codes and FLAME expression,
jaw, gaze and blinks) and, optionally, the head and upper-body motion. All driving videos are transferred onto one
canonical person, so any reference works with any driving video and any [camera](cameras.md).

## Example driving videos

The driving videos of the project page come with the examples (`python download.py examples`), in `examples/drivers/`.
They are NeRSemble performances from the frontal camera, 720×1280, 81 frames (3.4 s) at 24 fps:

| file | performance | NeRSemble subject |
|---|---|---|
| `disgust_happy.mp4` | disgust → happy | 125 |
| `tongue_out.mp4` | tongue out | 264 |
| `shout_laugh.mp4` | shout → laugh | 149 |
| `jaw.mp4` | jaw movements | 017 |
| `surprise_fear.mp4` | surprise → fear | 256 |
| `cheeks_nose.mp4` | cheeks and nose | 477 |

They are prepared like any video of your own (next section).

## Preparing a driving video

```bash
python prepare_driving.py --video examples/drivers/tongue_out.mp4 --name tongue_out
python render_conditioning.py --ref <NAME> --driving tongue_out --camera over_the_top_zoom --out_dir cache/cond/<RUN>
```

Use a video with one face, roughly frontal and well lit. `prepare_driving.py` (~17 min for an 81-frame video on an RTX A6000) runs:

1. frame extraction at 24 fps (`--fps`),
2. X-NeMo expression codes per frame (needs `weights/third_party/xnemo_motion_encoder.pth`),
3. pixel3dmm single-camera FLAME tracking for expression, jaw, gaze and eyelids (`--min_conf 0.25` for stylised
   faces, as for [reference images](reference_images.md)),
4. the transfer onto the canonical person.

The result is cached in `cache/driving/<DRV>/state.npz`; `render_conditioning.py --driving <DRV>` finds it by name.
`--force` recomputes it.

The full model uses the first 77 frames (3.2 s) of a driving video; the fast model uses all of it
([README, section 3](../README.md#3-long-videos-the-distilled-model)).

### Head and body motion

The driver's facial performance is always transferred. Head and body motion depend on `--motion`:

* `--motion take` (default): the canonical person's own head and upper-body motion from the shipped sequence
  `nersemble8` (below), looped back and forth to the video's length. `--take <K>` picks which of its eight performances
  provides that motion (default 1). The driver's own head rotation is not used.
* `--motion static`: the canonical person's head and body are held still.
* `--motion driver`: the driver's own head and upper-body motion, estimated with
  [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and transferred onto the canonical person's body
  (the pelvis stays in place, no walking; the driver's average posture of the head on the torso carries over).

`--motion driver` needs SAM 3D Body in its own conda env (see its repository) and one extra step, run from this
folder in that env:

```bash
python prepare_driving.py --video <DRIVER.mp4> --name <DRV>                         # frames, codes, FLAME tracking
conda run -n <SAM3D_ENV> python scripts/prepare_body_motion.py --name <DRV> --sam3d <SAM3D_REPO>   # [--no_fov]
python prepare_driving.py --video <DRIVER.mp4> --name <DRV> --motion driver        # reuses the first run's results
```

The third step replaces the first step's `state.npz` (same `--name`): render a take-mode conditioning before it if you
want both versions. On first use SAM 3D Body downloads code and models from the internet (its dinov3 backbone code
and the MoGe-2 field-of-view model, ~1.3 GB). SAM 3D Body needs ~11 GB of VRAM and ~1-2 s per frame (A100 / RTX 3090). On 11 GB cards use `--no_fov` (skips its
field-of-view estimator) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `--motion_sigma` (default 2
frames) smooths the estimated pose over time.

## The long driving sequence `nersemble8`

`weights/driving_bank/nersemble8.npz` comes with the weights. It is the 41 s driving sequence of the `distilled_*`
examples and of the fast model ([README, section 3](../README.md#3-long-videos-the-distilled-model)): eight NeRSemble
performances, each transferred onto the canonical person (NeRSemble subject 124), joined by 32-frame transitions;
1029 frames at 24 fps. It is already prepared, so it is used by name:

```bash
python render_conditioning.py --ref <NAME> --driving nersemble8 --camera sweep_41s --out_dir cache/cond/<RUN>
```

Each of the eight performances is called a *take*. `render_conditioning.py --take <K>` uses only take K (100 frames,
~4 s) instead of the whole sequence, e.g. for a full-model clip of one performance:
`--driving nersemble8 --take 3`. `--start <N>` skips the first N frames and `--max_frames <N>` keeps at most N.

| take | frames | NeRSemble subject | performance |
|---|---|---|---|
| 1 | 0–100 | 124 | disgust → happy |
| 2 | 133–233 | 125 | disgust → happy |
| 3 | 266–366 | 074 | jaw |
| 4 | 399–499 | 252 | angry → sad |
| 5 | 532–632 | 320 | lips |
| 6 | 665–764 | 061 | cheeks and nose |
| 7 | 797–896 | 393 | mouth |
| 8 | 929–1028 | 226 | disgust → happy |

The labels are also in `weights/driving_bank/nersemble8.takes.json`. Take 1 is also the canonical person's default
head and body motion for your own driving videos (`--motion take`, above).

## Expression codes only

The X-NeMo codes of any video, e.g. for the [disentangler](python.md#expression-disentangler):

```bash
python scripts/extract_xnemo_codes.py --video <VIDEO.mp4> --out <CODES.npy>       # [T, 512]
```
