# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
"""WAN-Animate-14B pipeline for video-diffusion-avatars (trimmed from DiffSynth-Studio).

Only what the two released recipes execute is kept:
  * full model  : bidirectional DiT + X-NeMo animate adapter, one 77-frame window per call,
                  ODE sampler, joint guidance (2 passes per step); the fast model's teacher prefix runs the
                  same path with nested expr/id/graft guidance (4 passes per step);
  * fast model  : the causal Self-Forcing student, streamed block-by-block on a KV cache
                  (`kv_stream`, see _kv_stream_window), SDE sampler, no guidance.
Everything here (pipeline units, __call__, _kv_stream_window, model_fn_wan_video) is reduced to the
branches these two recipes execute; the numerics are unchanged by the trimming.
"""
import torch
from PIL import Image
from einops import rearrange
from typing import Optional, Union
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_animate_adapter import WanAnimateAdapter



class WanVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler()
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.animate_adapter: WanAnimateAdapter = None
        self.in_iteration_models = ("dit", "animate_adapter")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateGraftLatents(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_AnimateRefOOT(),
            WanVideoUnit_IdWeakRef(),      # after Inpaint/RefOOT: needs the final y; inert unless cfg_scale_id != 1
            WanVideoUnit_ExprCodes(),
        ]
        self.model_fn = model_fn_wan_video

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    ):
        # The original .pth files of the text encoder, CLIP and VAE are read from DiffSynth's converted safetensors
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
            "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
        }
        for model_config in model_configs:
            if model_config.origin_file_pattern is None or model_config.model_id is None:
                continue
            if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs)
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            raise ValueError("Wan2.2-Animate-14B has a single DiT; got two (wrong model files?)")
        pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        # appearance reference (the out-of-timeline latent slot 0)
        input_image: Image.Image = None,
        # 2D skeleton video, graft (FLAME head + body; = pose_proxy.mp4, the "proxy render" of the docs) video,
        # generate-region fill/mask videos.
        # Out-of-timeline reference contract: latent slot 0 = VAE(input_image) (never re-pinned during
        # sampling, as trained), stripped before decoding, so the output has num_frames - 4 frames
        # aligned 1:1 with pose/graft/codes.
        animate_pose_video: list[Image.Image] = None,
        animate_graft_video: list[Image.Image] = None,
        animate_inpaint_video: list[Image.Image] = None,
        animate_mask_video: list[Image.Image] = None,
        # RAW X-NeMo expression codes [T, 512] (the frozen in-graph disentangler encodes them)
        expr_codes: torch.Tensor = None,
        seed: int = None,
        rand_device: str = "cpu",
        height: int = 1024,
        width: int = 576,
        num_frames: int = 81,
        # guidance. nested: eps = base + s_id*(nega-base) + cfg_scale*(posi-nega) (+ graft term);
        # joint (cfg_joint): eps = base + cfg_scale_id*(posi-base), 2 passes per step.
        # base = weak (blurred) reference with the expression dropped (the identity-dropout condition seen in training).
        cfg_scale: float = 1.0,
        cfg_scale_id: float = 1.0,
        id_weak_image: Image.Image = None,
        cfg_scale_graft: float = 1.0,
        cfg_joint: bool = False,
        # causal student: a mutable state dict shared across calls (see _kv_stream_window)
        kv_stream: Optional[dict] = None,
        return_latents: bool = False,
        sampler: str = "ode",
        num_inference_steps: int = 4,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)
        inputs_posi = {"prompt": prompt, "num_inference_steps": num_inference_steps}
        inputs_nega = {"negative_prompt": negative_prompt, "num_inference_steps": num_inference_steps}
        inputs_shared = {
            "input_image": input_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_scale_id": cfg_scale_id, "id_weak_image": id_weak_image,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "animate_pose_video": animate_pose_video, "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video, "animate_graft_video": animate_graft_video,
            "expr_codes": expr_codes,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # weak-identity branch tensors are consumed by the base pass only
        _y_weak = inputs_shared.pop("y_weak", None)
        _clip_weak = inputs_shared.pop("clip_feature_weak", None)
        _ffl_weak = inputs_shared.pop("first_frame_latents_weak", None)
        inputs_shared.pop("id_weak_image", None)
        inputs_shared.pop("cfg_scale_id", None)
        _id_cfg = (_y_weak is not None and float(cfg_scale_id) != 1.0)
        _graft_cfg = (float(cfg_scale_graft) != 1.0 and inputs_shared.get("graft_latents") is not None)
        if cfg_joint and not _id_cfg:
            raise ValueError("cfg_joint needs the weak-identity base pass: set cfg_scale_id != 1 "
                             "(it is the joint scale) and pass id_weak_image.")

        models = {name: getattr(self, name) for name in self.in_iteration_models}
        if kv_stream is not None:
            if sampler != "sde" or cfg_scale != 1.0 or _id_cfg or _graft_cfg or cfg_joint:
                raise ValueError("kv_stream is the distilled causal student's path: sampler='sde', "
                                 "no guidance (cfg_scale/cfg_scale_id/cfg_scale_graft == 1).")
            return self._kv_stream_window(kv_stream, models, inputs_shared, inputs_posi, seed, rand_device)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            def _base_pass():
                # negative conditioning with the identity degraded the way identity dropout degraded it
                # in training: y ref slot, CLIP, and slot 0 of its own latent view
                _lat_weak = inputs_shared["latents"].clone()
                _lat_weak[:, :, 0:1] = _ffl_weak.to(_lat_weak.dtype)
                _over = {"latents": _lat_weak, "y": _y_weak}
                if _clip_weak is not None:
                    _over["clip_feature"] = _clip_weak
                return self.model_fn(**models, **{**inputs_shared, **_over}, **inputs_nega, timestep=timestep)

            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_joint:
                noise_pred_base = _base_pass()
                noise_pred = noise_pred_base + cfg_scale_id * (noise_pred_posi - noise_pred_base)
            elif cfg_scale != 1.0 or _id_cfg:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                if _id_cfg:
                    noise_pred_base = _base_pass()
                    noise_pred = (noise_pred_base
                                  + cfg_scale_id * (noise_pred_nega - noise_pred_base)
                                  + cfg_scale * (noise_pred_posi - noise_pred_nega))
                else:
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            if _graft_cfg:
                # leave-one-out graft guidance at the full positive operating point
                # (graft_latents=None is the trained graft-absent state)
                noise_pred_ng = self.model_fn(
                    **models, **{**inputs_shared, "graft_latents": None}, **inputs_posi, timestep=timestep)
                noise_pred = noise_pred + (cfg_scale_graft - 1.0) * (noise_pred_posi - noise_pred_ng)

            if sampler == "sde":
                # x0-prediction + fresh-noise re-noising (the distilled student's sampler)
                _sg = float(self.scheduler.sigmas[progress_id])
                _sg_next = (float(self.scheduler.sigmas[progress_id + 1])
                            if progress_id + 1 < len(self.scheduler.sigmas) else 0.0)
                _x0 = inputs_shared["latents"] - _sg * noise_pred
                if _sg_next > 0:
                    _eps = self.generate_noise(
                        _x0.shape, seed=(None if seed is None else int(seed) + 1000 + progress_id),
                        rand_device=rand_device, torch_dtype=_x0.dtype, device=_x0.device)
                    inputs_shared["latents"] = (1 - _sg_next) * _x0 + _sg_next * _eps
                else:
                    inputs_shared["latents"] = _x0
            else:
                inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # strip the out-of-timeline reference slot
        inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]
        if return_latents:
            return inputs_shared["latents"]
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        return video




def _kv_stream_window(self, st, models, inputs_shared, inputs_posi, seed, rand_device):
    """One window of Self-Forcing streaming on the persistent KV cache `st`:
        st = {"chunk": latent slots per block, "window": trained video slots (20), "caches": None,
              "sink": video slots pinned after the reference, "ref_dup": reference boost k,
              "prefill": teacher latents for the first slots (or None)}
    (kv_n / n_ctx / blocks_done are managed here). Cache = [reference sink | <= window video slots].
    The first window runs exactly like training (positions 1..window, growing). Afterwards, before
    each block the oldest slots are evicted and the remaining context re-rotated so it sits at
    positions 1..window-chunk with the new block at window-chunk+1..window: every block is generated
    as the LAST block of a training clip (rolling RoPE + reference attention sink, cf. Live Avatar /
    Rolling Forcing). Noise seeds: step j of global block `bd` uses seed + 1000 + 100*bd + j (full-shape
    noise, sliced to the block), so a run is reproducible for a given seed."""
    from .causal_sampler import chunk_bounds, kv_evict_shift
    dit = self.dit
    lat = inputs_shared["latents"]
    ffl = inputs_shared["first_frame_latents"].to(lat.dtype)
    B, C, T, H, W = lat.shape
    tokf = (H // 2) * (W // 2)
    chunk = int(st.get("chunk", 4))
    window = int(st.get("window", T - 1))
    n_heads = dit.blocks[0].self_attn.num_heads
    head_dim = dit.blocks[0].self_attn.head_dim
    sigmas = [float(s) for s in self.scheduler.sigmas] + [0.0]
    timesteps = self.scheduler.timesteps.to(dtype=self.torch_dtype, device=self.device)

    def fwd(lat_b, tvec, s_rope, s_cond, mode):
        inp = {**inputs_shared, "latents": lat_b,
               "y": inputs_shared["y"][:, :, s_cond:s_cond + lat_b.shape[2]],
               "kv_caches": st["caches"], "kv_mode": mode, "kv_n": int(st["kv_n"]),
               "frame_offset": int(s_rope), "cond_offset": int(s_cond)}
        return self.model_fn(**models, **inp, **inputs_posi, timestep=tvec)

    with torch.no_grad():
        if st.get("caches") is None:
            st["caches"] = [dict(k=None, v=None, mode=None) for _ in dit.blocks]
            st["kv_n"], st["n_ctx"], st["blocks_done"] = 0, 0, 0
            fwd(ffl, torch.zeros((B, 1), dtype=self.torch_dtype, device=self.device), 0, 0, "fill")
            st["kv_n"] = tokf
            # reference boost: +log(k) on every reference token's attention logit, for every later
            # query (the same math as k identical copies of the reference K/V, without the memory)
            dup = int(st.get("ref_dup", 1))
            if dup > 1:
                import math as _m
                for c in st["caches"]:
                    c["ref_bias_tok"], c["ref_bias"] = tokf, _m.log(dup)
                st["ref_slots"] = 1
        outs = []
        import time as _time
        torch.cuda.synchronize()
        _t_start, _t_blocks = _time.time(), []            # per-block wall time (speed / latency report)
        for ci, (s, e) in enumerate(chunk_bounds(T, chunk, 1)):
            nb = e - s
            over = st["n_ctx"] + nb - window
            sink = int(st.get("sink", 0))                  # video slots pinned after the reference
            if over > 0:                                   # roll: evict the oldest `over` NON-sink slots
                kv_evict_shift(st["caches"], (int(st.get("ref_slots", 1)) + sink) * tokf, over * tokf, -over,
                               dit.freqs[0], n_heads, head_dim)
                st["kv_n"] -= over * tokf
                st["n_ctx"] -= over
            s_rope = st["n_ctx"] + 1
            bd = int(st["blocks_done"])
            pre = st.get("prefill")
            if pre is not None and e - 1 <= int(pre.shape[2]):
                # teacher-quality prefix: these slots were generated by the bidirectional teacher
                x0 = pre[:, :, s - 1:e - 1].to(device=lat.device, dtype=lat.dtype)
            else:
                zc = lat[:, :, s:e]
                n_run = len(timesteps)
                for j in range(n_run):
                    tvec = torch.full((B, nb), float(timesteps[j]), dtype=self.torch_dtype, device=self.device)
                    last = j + 1 == n_run
                    v = fwd(zc, tvec, s_rope, s, "query")
                    x0 = zc.double() - sigmas[j] * v.double()
                    if not last:
                        s_next = sigmas[j + 1]
                        eps = self.generate_noise(
                            lat.shape, seed=(None if seed is None else int(seed) + 1000 + 100 * bd + j),
                            rand_device=rand_device, torch_dtype=lat.dtype, device=lat.device)[:, :, s:e]
                        zc = ((1.0 - s_next) * x0 + s_next * eps.double()).to(lat.dtype)
                x0 = x0.to(lat.dtype)
            outs.append(x0)
            fwd(x0, torch.zeros((B, nb), dtype=self.torch_dtype, device=self.device), s_rope, s, "fill")
            st["kv_n"] += nb * tokf
            st["n_ctx"] += nb
            st["blocks_done"] = bd + 1
            torch.cuda.synchronize()
            _t_blocks.append(_time.time())
        if _t_blocks:
            _dt = [b - a for a, b in zip([_t_start] + _t_blocks[:-1], _t_blocks)]
            _steady = _dt[1:] if len(_dt) > 1 else _dt
            _m = sum(_steady) / len(_steady)
            print(f"[kv-timing] {len(_dt)} blocks; first block (13 frames) ready after {_dt[0]:.2f}s; "
                  f"steady {_m:.2f}s per 4-latent block (16 frames) = {16 / _m:.2f} fps latent generation "
                  f"(excl. conditioning encode + VAE decode); total {_t_blocks[-1] - _t_start:.1f}s", flush=True)
    return torch.cat(outs, dim=2)


WanVideoPipeline._kv_stream_window = _kv_stream_window



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("latents",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        return {"latents": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt) -> dict:
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "height", "width"),
            output_params=("clip_feature",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, height, width):
        if input_image is None or pipe.image_encoder is None:
            return {}
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}



class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}



class WanVideoUnit_AnimateGraftLatents(PipelineUnit):
    """Proxy-render ("graft") conditioning, pose_proxy.mp4 (FLAME head on the MHR body): VAE-encode it exactly
    like the skeleton pose video; the adapter adds it to the pose tokens through a zero-initialised patch
    embedding."""
    def __init__(self):
        super().__init__(
            input_params=("animate_graft_video", "tiled", "tile_size", "tile_stride"),
            output_params=("graft_latents",)
        )

    def process(self, pipe: WanVideoPipeline, animate_graft_video, tiled, tile_size, tile_stride):
        if animate_graft_video is None:
            return {}
        animate_graft_video = pipe.preprocess_video(animate_graft_video)
        graft_latents = pipe.vae.encode(animate_graft_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"graft_latents": graft_latents}



class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("y",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device=get_device_type()):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}



class WanVideoUnit_AnimateRefOOT(PipelineUnit):
    """Out-of-timeline reference: the ref latent that WanVideoUnit_AnimateInpaint already VAE-encoded
    into y's slot 0 (channels 4:20) becomes `first_frame_latents` (the KV cache's reference sink in the
    streaming path). Together with the end-of-denoise latent strip in __call__ this restores the stock
    Animate contract: output = num_frames-4 real frames, pose/code latent k conditions exactly the
    frames that output latent k decodes."""
    def __init__(self):
        super().__init__(
            input_params=("y",),
            output_params=("first_frame_latents",),
        )

    def process(self, pipe: WanVideoPipeline, y):
        assert y is not None, (
            "the out-of-timeline reference requires the AnimateInpaint conditioning "
            "(animate_inpaint_video + animate_mask_video + input_image).")
        return {"first_frame_latents": y[:, 4:, 0:1]}  # [B, 16, 1, h, w] = VAE(input_image)



class WanVideoUnit_IdWeakRef(PipelineUnit):
    """Weak-identity conditioning branch for the guided base pass.

    Consumes `id_weak_image` (the DEGRADED reference -- Gaussian-blurred by the caller with the
    SAME sigma as the identity-dropout augmentation in training) and builds the three tensors the base pass
    swaps in: `y_weak` (y with the ref slot's 16 latent channels re-encoded from the weak
    image; the 4 mask channels stay), `clip_feature_weak`, and `first_frame_latents_weak`
    (the weak ref latent, pinned into slot 0 of the base pass's PRIVATE latent view -- under
    identity dropout in training the blurred ref filled slot 0 too, so the base pass must see it there,
    while the shared denoise trajectory stays pinned to the REAL ref).

    Inert unless both `id_weak_image` is given and `cfg_scale_id != 1`.
    """
    def __init__(self):
        super().__init__(
            input_params=("id_weak_image", "cfg_scale_id", "y", "clip_feature",
                          "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y_weak", "clip_feature_weak", "first_frame_latents_weak")
        )

    def process(self, pipe: WanVideoPipeline, id_weak_image, cfg_scale_id, y, clip_feature,
                height, width, tiled, tile_size, tile_stride):
        if id_weak_image is None or cfg_scale_id is None or float(cfg_scale_id) == 1.0:
            return {}
        if y is None:
            raise ValueError("cfg_scale_id needs the animate reference conditioning "
                             "(animate_inpaint_video + animate_mask_video + input_image).")
        weak_pix = pipe.preprocess_video([id_weak_image])
        weak_lat = pipe.vae.encode(weak_pix, device=pipe.device, tiled=tiled,
                                   tile_size=tile_size, tile_stride=tile_stride
                                   ).to(dtype=y.dtype, device=y.device)
        y_weak = y.clone()
        y_weak[:, 4:, 0:1] = weak_lat
        out = {"y_weak": y_weak, "first_frame_latents_weak": weak_lat}
        if clip_feature is not None and pipe.image_encoder is not None:
            img = pipe.preprocess_image(id_weak_image.resize((width, height))).to(pipe.device)
            out["clip_feature_weak"] = pipe.image_encoder.encode_image([img]).to(
                dtype=clip_feature.dtype, device=clip_feature.device)
        return out



class WanVideoUnit_ExprCodes(PipelineUnit):
    """Per-frame RAW X-NeMo expression codes ([T, 512] or [B, T, 512], tensor or numpy): batched,
    cast, trimmed to the num_frames - 4 frames the out-of-timeline reference leaves for the video
    (the face encoder's 4x temporal downsample + the reference-slot pad must land exactly on the
    latent frame count), and zeroed on the CFG negative side."""
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("expr_codes",),
            output_params=("expr_codes",),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        expr_codes = inputs_shared.pop("expr_codes", None)
        if expr_codes is None:
            return inputs_shared, inputs_posi, inputs_nega
        if not torch.is_tensor(expr_codes):
            expr_codes = torch.as_tensor(expr_codes)
        if expr_codes.dim() == 2:
            expr_codes = expr_codes.unsqueeze(0)
        expr_codes = expr_codes.to(device=pipe.device, dtype=pipe.torch_dtype)
        target = inputs_shared["num_frames"] - 4
        if expr_codes.shape[1] > target:
            expr_codes = expr_codes[:, :target]
        inputs_posi["expr_codes"] = expr_codes
        inputs_nega["expr_codes"] = torch.zeros_like(expr_codes)
        return inputs_shared, inputs_posi, inputs_nega



def model_fn_wan_video(
    dit: WanModel,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    pose_latents=None,
    graft_latents=None,
    expr_codes: Optional[torch.Tensor] = None,
    # Self-Forcing KV cache (block-only forwards, models/wan_video_dit.py SelfAttention):
    # kv_caches = one dict per DiT block ({"k","v"} post-RoPE, grown by "fill" forwards); kv_mode
    # "query" (attend to cache[:kv_n] + own tokens) or "fill" (same, then append own K/V; run at
    # t=0); frame_offset = absolute latent-frame index of latents[:, :, 0] (RoPE);
    # cond_offset = row of the window's conditioning (pose rows / motion tokens), None = frame_offset.
    kv_caches: Optional[list] = None,
    kv_mode: Optional[str] = None,
    kv_n: int = 0,
    frame_offset: int = 0,
    cond_offset: Optional[int] = None,
    **kwargs,
):
    # Timestep
    if timestep.dim() == 2:
        # per-latent-frame timestep [B, T_lat] (the KV-streaming student: cache fills at t=0,
        # denoising blocks at t), expanded per patch token
        assert timestep.shape[1] == latents.shape[2], \
            f"per-frame timestep length {timestep.shape[1]} != latent frames {latents.shape[2]}"
        _tok_per_frame = latents.shape[3] * latents.shape[4] // 4
        _t_tok = timestep.repeat_interleave(_tok_per_frame, dim=1).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, _t_tok).unsqueeze(0).to(latents.dtype))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image embedding (VAE channel concat + CLIP tokens)
    if y is not None:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    x = dit.patchify(x)

    # Animate: pose (+ graft) patch embedding and X-NeMo motion tokens (WanAnimateAdapterXNemo)
    use_xnemo_animate = pose_latents is not None and expr_codes is not None and animate_adapter is not None
    if use_xnemo_animate:
        x, motion_vec = animate_adapter.after_patch_embedding_xnemo(
            x, pose_latents, expr_codes, graft_latents=graft_latents,
            frame_offset=(frame_offset if cond_offset is None else cond_offset))

    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    # frame_offset: absolute temporal RoPE positions for a KV-cached block forward (0 = unchanged)
    freqs = torch.cat([
        dit.freqs[0][frame_offset:frame_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    if kv_caches is not None:
        assert kv_mode in ("query", "fill") and len(kv_caches) == len(dit.blocks), \
            f"kv_mode {kv_mode!r}, {len(kv_caches)} caches for {len(dit.blocks)} blocks"
        for _c in kv_caches:
            _c["mode"] = kv_mode

    for block_id, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs, (kv_caches[block_id] if kv_caches is not None else None), kv_n)
        if use_xnemo_animate:
            # face-fuser cross-attention to the motion tokens (fires on block_id % 5 == 0)
            x = animate_adapter.after_transformer_block(block_id, x, motion_vec)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
