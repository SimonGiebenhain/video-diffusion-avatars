"""Load the WAN-Animate-14B pipeline with a video-diffusion-avatars adapter (and optionally the
distilled student). The order of operations matters for reproducing the released outputs bit for bit
and is commented where it does."""
import os

import torch

from . import paths

ANIMATE_ID = "Wan-AI/Wan2.2-Animate-14B"
DEFAULT_PROMPT = "a studio portrait video of a person performing facial expressions"


def _install_text_cache(pipe, cache_path, prompts):
    """Replace the umt5 prompt encoder by cached embeddings (the prompts are constant)."""
    from vda.third_party.diffsynth.pipelines import wan_video as _wv
    cache = torch.load(cache_path, map_location="cpu")
    missing = [p for p in prompts if p not in cache]
    if missing:
        raise SystemExit(f"text-embedding cache {cache_path} lacks prompt(s) {missing}; "
                         f"run `python scripts/add_prompt.py --prompt ...` once (loads umt5-xxl)")
    frozen = dict(cache)

    def _cached_process(self, pipe_, prompt):
        return {"context": frozen[prompt].to(device=pipe_.device, dtype=pipe_.torch_dtype)}

    _wv.WanVideoUnit_PromptEmbedder.process = _cached_process


def load_pipeline(adapter_path, prompt=DEFAULT_PROMPT, negative_prompt=None,
                  disentangler_path=None, text_cache_path=None, device="cuda"):
    """-> WanVideoPipeline with the X-NeMo adapter (translator + graft branch) loaded, every model
    resident on `device` (stream DiT blocks from host memory with enable_block_offload)."""
    paths.setup_base_model_env()
    from vda.third_party.diffsynth.core import load_state_dict
    from vda.third_party.diffsynth.models.wan_video_animate_adapter_xnemo import WanAnimateAdapterXNemo, build_translator_head
    from vda.third_party.diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    negative_prompt = prompt if negative_prompt is None else negative_prompt
    text_cache_path = text_cache_path or paths.weight(paths.TEXT_CACHE)
    disentangler_path = disentangler_path or paths.weight(paths.DISENTANGLER)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[
            ModelConfig(model_id=ANIMATE_ID, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=ANIMATE_ID, origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id=ANIMATE_ID, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )
    _install_text_cache(pipe, text_cache_path, [prompt, negative_prompt])
    assert pipe.animate_adapter is not None, "Wan2.2-Animate-14B must bundle the animate_adapter"

    # Keep the loaded stock weights (motion_encoder, pose_patch_embedding, ...) and attach the new
    # modules BEFORE loading the checkpoint: load_state_dict(strict=False) only fills modules that exist.
    ad = pipe.animate_adapter
    ad.__class__ = WanAnimateAdapterXNemo
    ad.translator = build_translator_head(disentangler_path, device)
    _ppe = ad.pose_patch_embedding
    _ge = torch.nn.Conv3d(_ppe.in_channels, _ppe.out_channels, kernel_size=_ppe.kernel_size, stride=_ppe.stride)
    torch.nn.init.zeros_(_ge.weight)
    torch.nn.init.zeros_(_ge.bias)
    ad.graft_patch_embedding = _ge.to(device=_ppe.weight.device, dtype=_ppe.weight.dtype)

    # fp32 load: the translator is an fp32 module (bf16 modules downcast on copy anyway)
    trained = load_state_dict(adapter_path, torch_dtype=torch.float32, device=device)
    trained = {k: v for k, v in trained.items() if not k.startswith("gates.")}   # unused (ungated residual)
    missing, unexpected = ad.load_state_dict(trained, strict=False)
    if unexpected:
        raise SystemExit(f"adapter {adapter_path}: unexpected keys {unexpected[:5]} -- not a released checkpoint?")
    need = [k for k, _ in ad.named_parameters()
            if k.startswith(("translator.f.", "graft_patch_embedding.", "face_adapter.", "face_encoder."))]
    lost = [k for k in need if k not in trained]
    if lost:
        raise SystemExit(f"adapter {adapter_path} does not provide {lost[:5]} ({len(lost)} tensors)")
    print(f"[vda] adapter {os.path.basename(adapter_path)}: {len(trained)} tensors "
          f"(kept from base/disentangler: {len(missing)})", flush=True)
    pipe.vda_prompts = (prompt, negative_prompt)
    return pipe


def load_student_lora(pipe, lora_path, alpha=1.0):
    """Merge the distilled student's DiT LoRA into the DiT weights and inject its face-fuser LoRA
    (PEFT, alpha = rank, as trained). Irreversible for this process: call it AFTER anything that
    must run on the teacher (the teacher prefix)."""
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    sd = load_file(lora_path)
    dit_sd = {}
    for k, v in sd.items():
        if "lora_" not in k or k.startswith("pipe.animate_adapter"):
            continue
        if k.startswith("pipe.dit."):
            k = k[len("pipe.dit."):]
        dit_sd[k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")] = v
    if not dit_sd:
        raise SystemExit(f"{lora_path}: no DiT LoRA tensors")
    pipe.load_lora(pipe.dit, state_dict=dit_sd, alpha=alpha)
    fuser = {k[len("pipe.animate_adapter."):]: v for k, v in sd.items()
             if k.startswith("pipe.animate_adapter.") and ".lora_" in k}
    if fuser:
        rank = int(next(v for k, v in fuser.items() if ".lora_A." in k).shape[0])
        targets = r"fuser_blocks\.[0-9]+\.(linear1_kv|linear1_q|linear2)"
        inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, target_modules=targets),
                                pipe.animate_adapter.face_adapter)
        for _, mod in pipe.animate_adapter.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
                w = mod.base_layer.weight
                for sub in (mod.lora_A, mod.lora_B):
                    for p in sub.parameters():
                        p.data = p.data.to(device=w.device, dtype=w.dtype)
        _, unexp = pipe.animate_adapter.load_state_dict(fuser, strict=False)
        live = {k for k, _ in pipe.animate_adapter.named_parameters() if ".lora_" in k}
        if unexp or set(fuser) != live:
            raise SystemExit(f"fuser LoRA mismatch: {len(unexp)} unexpected, {len(live - set(fuser))} not in file")
    print(f"[vda] student LoRA {os.path.basename(lora_path)}: {len(dit_sd)} DiT + {len(fuser)} fuser tensors", flush=True)


def enable_block_offload(pipe, n_blocks):
    """Keep the LAST n DiT blocks in pinned host memory and stream each to the GPU for its own
    forward (same weights, so the same output; lowers the peak VRAM by ~0.8 GB per block (bf16)). Call once
    the weights are final: after load_pipeline for the full model, after load_student_lora for the
    fast one (the pinned copies must carry the merged weights)."""
    if n_blocks <= 0:
        return
    blocks = list(pipe.dit.blocks)[-n_blocks:]
    dev = next(pipe.dit.parameters()).device
    for b in blocks:
        host = {}
        for p in b.parameters():
            host[p] = p.data.to("cpu").pin_memory()
            p.data = host[p]

        def _pre(mod, inp, _host=host):
            for p, h in _host.items():
                p.data = h.to(dev, non_blocking=True)

        def _post(mod, inp, out, _host=host):
            for p, h in _host.items():
                p.data = h
            return out
        b.register_forward_pre_hook(_pre)
        b.register_forward_hook(_post)
    torch.cuda.empty_cache()
    print(f"[vda] {len(blocks)}/{len(pipe.dit.blocks)} DiT blocks streamed from host memory; "
          f"resident {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
