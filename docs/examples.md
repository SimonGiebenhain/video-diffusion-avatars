# Examples

The 25 examples from the [project page](https://simongiebenhain.github.io/video-diffusion-avatars/), with their conditioning precomputed: the reference image,
the proxy render, the skeleton and the expression codes. They run without any preprocessing.

```bash
python download.py examples                                              # all 25 + the driving videos, ~220 MB
python download.py examples --names teaser_swimmer distilled_redbangs    # or just a few
```

Each `examples/<NAME>/` holds `ref.png` (720×1280 reference), `pose_proxy.mp4` (proxy render), `pose_skel.mp4`
(skeleton), `codes.npy` (expression codes) and `meta.json` (subject, driver, camera, model). Run one with
```bash
python generate.py --cond examples/<NAME> --out outputs/<NAME>.mp4
```
and the `distilled_*` ones with the fast model:
```bash
python generate.py --model fast --cond examples/<NAME> --out outputs/<NAME>.mp4 --deflicker
```

All full-model examples are 77 frames (3.2 s) except `control_neck_red_hair_teen` (71 conditioning frames, 69
generated: the full model generates 4k+1 frames). The expression drivers are NeRSemble performances; their videos are in `examples/drivers/` ([driving.md](driving.md)).
The references are
shown larger, with tips for your own, in [reference_images.md](reference_images.md); the camera presets in
[cameras.md](cameras.md).

## Teaser: expressions and camera

| reference | example | expression | camera |
|---|---|---|---|
| <img src="refs/swimmer.jpg" width="72"> | `teaser_swimmer`<br>swimmer with goggles | disgust → happy | `over_the_top_zoom_55` |
| <img src="refs/gele.jpg" width="72"> | `teaser_gele`<br>woman in a gele headwrap | tongue out | over the top + zoom at 75 % amplitude (not a preset; closest: `over_the_top_zoom_75m`, mirrored) |
| <img src="refs/greenhair.jpg" width="72"> | `teaser_greenhair`<br>green hair | shout → laugh | `over_the_top_zoom_55` |
| <img src="refs/hijab.jpg" width="72"> | `teaser_hijab`<br>woman in a hijab | disgust → happy | `over_the_top_zoom_55` |
| <img src="refs/cartoon.jpg" width="72"> | `teaser_cartoon`<br>cartoon portrait | tongue out | over the top + zoom at 75 % amplitude (not a preset; closest: `over_the_top_zoom_75m`, mirrored) |
| <img src="refs/jgrandma.jpg" width="72"> | `teaser_grandma`<br>grandmother with glasses | jaw movements | `over_the_top_zoom` |
| <img src="refs/vangogh.jpg" width="72"> | `teaser_vangogh`<br>Van Gogh self-portrait | the photo's own expression | `ladder_zoom` |

## Camera control

The photo's expression is kept; only the camera moves.

| reference | example | camera |
|---|---|---|
| <img src="refs/tortoise.jpg" width="72"> | `control_camera_tortoise`<br>woman with tortoise-shell sunglasses | `over_the_top_zoom` |
| <img src="refs/curlyboy.jpg" width="72"> | `control_camera_curlyboy`<br>animated boy with curly hair | `ladder_zoom_m` |
| <img src="refs/refraction.jpg" width="72"> | `control_camera_refraction`<br>man with black-framed glasses | `over_the_top_zoom` |

## Neck pose control

Static camera; the head moves on the neck while the expression stays that of the photo.

| reference | example | neck | frames |
|---|---|---|---|
| <img src="refs/neon_earring.jpg" width="72"> | `control_neck_neon_earring`<br>young man with long hair in neon light | head nods yes / no | 77 |
| <img src="refs/red_hair_teen.jpg" width="72"> | `control_neck_red_hair_teen`<br>teenager with red hair | head nods yes / no | 71 |
| <img src="refs/straw_hat.jpg" width="72"> | `control_neck_straw_hat`<br>woman with a straw hat | head rolls | 77 |
| <img src="refs/beanie_locs.jpg" width="72"> | `control_neck_beanie_locs`<br>woman with a beanie and locs | head turns | 77 |

## Expression control while the camera moves

| reference | example | expression | camera |
|---|---|---|---|
| <img src="refs/swimmer.jpg" width="72"> | `control_expr_swimmer`<br>swimmer with goggles | surprise → fear | `over_the_top_zoom_55` |
| <img src="refs/blinds.jpg" width="72"> | `control_expr_blinds`<br>woman with glasses in striped light | jaw movements | `ladder_zoom` |
| <img src="refs/gele.jpg" width="72"> | `control_expr_gele`<br>woman wearing a gele | cheeks and nose | `ladder_zoom` |

## Long videos: the distilled model

All eight use the whole shipped driving sequence (8 NeRSemble takes, 1029 frames, 41 s at 24 fps, see
[driving.md](driving.md)) and the `sweep_41s` camera, which swings between ±65° azimuth. See
[README, section 3](../README.md#3-long-videos-the-distilled-model).

| reference | example |
|---|---|
| <img src="refs/redbangs.jpg" width="72"> | `distilled_redbangs`<br>red bangs |
| <img src="refs/greenhair.jpg" width="72"> | `distilled_greenhair`<br>green hair |
| <img src="refs/vangogh.jpg" width="72"> | `distilled_vangogh`<br>Van Gogh portrait |
| <img src="refs/frida.jpg" width="72"> | `distilled_frida`<br>Frida Kahlo painting |
| <img src="refs/marble.jpg" width="72"> | `distilled_marble`<br>marble bust |
| <img src="refs/watchmaker.jpg" width="72"> | `distilled_watchmaker`<br>3D-animated watchmaker |
| <img src="refs/cgrandma.jpg" width="72"> | `distilled_cgrandma`<br>cartoon grandma |
| <img src="refs/refraction.jpg" width="72"> | `distilled_refraction`<br>man with black-framed glasses |
