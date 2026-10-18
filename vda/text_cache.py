"""Add prompt embeddings to weights/text_emb_cache.pt (only needed for a NEW prompt; loads umt5-xxl, 11 GB).

    python download.py base --with_text_encoder
    python scripts/add_prompt.py --prompt "a studio portrait video of a person performing facial expressions"

The released models were trained with ONE fixed prompt (the default above) and use it as the negative
prompt too; other prompts are untested.
"""
import argparse
import os

import torch

from . import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", required=True)
    a = ap.parse_args()
    paths.setup_base_model_env()
    from vda.third_party.diffsynth.pipelines import wan_video as wv
    from vda.third_party.diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    cache_path = paths.weights_dir() / paths.TEXT_CACHE
    cache = torch.load(cache_path, map_location="cpu") if cache_path.exists() else {}
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"))
    emb = wv.WanVideoUnit_PromptEmbedder()
    with torch.no_grad():
        for p in a.prompt:
            cache[p] = emb.encode_prompt(pipe, p).to("cpu")
    os.makedirs(cache_path.parent, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[vda] text cache {cache_path}: {len(cache)} prompts")


if __name__ == "__main__":
    main()
