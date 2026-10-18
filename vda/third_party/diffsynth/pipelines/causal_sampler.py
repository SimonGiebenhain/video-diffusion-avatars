# video-diffusion-avatars code, CC BY-NC-SA 4.0 (LICENSE at the repository root). It builds on the
# DiffSynth-Studio code in this folder (Apache-2.0, vda/third_party/diffsynth/LICENSE).
"""Block bookkeeping for the KV-cached Self-Forcing student (`WanVideoPipeline._kv_stream_window`):
the latent-slot ranges of the generated blocks and the rolling-window eviction of the KV cache."""
import torch


def chunk_bounds(n_slots, chunk, prefix):
    """[(start, end)] latent-slot ranges of the generated chunks.

    A short remainder is absorbed into the LAST chunk instead of forming its own: a lone tail slot (no
    neighbours in its chunk, conditioned on a full clean context) is far out of distribution and degrades
    visibly."""
    chunk = int(chunk)
    assert chunk > 0, "causal chunk must be positive"
    prefix = int(prefix)
    out = [(s, min(s + chunk, n_slots)) for s in range(prefix, n_slots, chunk)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < chunk:      # absorb the remainder
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return out


def kv_evict_shift(caches, keep_tok, drop_tok, delta_frames, freqs_f, num_heads, head_dim):
    """Rolling KV window (serving): keep the first keep_tok tokens (the reference sink), drop the next
    drop_tok (the oldest video slots) and rotate the temporal RoPE of every remaining video token by
    delta_frames. K is cached POST-RoPE at absolute positions and rotations compose, so a uniform
    shift is one complex multiply on the temporal components (the first head_dim - 2*(head_dim//3)
    real dims of each head; `freqs_f` = dit.freqs[0], polar(1, p*theta_i) per frame p)."""
    delta_frames = int(delta_frames)
    if drop_tok == 0 and delta_frames == 0:
        return
    n_f = freqs_f.shape[1]
    rot = freqs_f[abs(delta_frames)].to(torch.complex128)
    if delta_frames < 0:
        rot = rot.conj()
    for c in caches:
        k, v = c["k"], c["v"]
        if k is None:
            continue
        k = torch.cat([k[:, :keep_tok], k[:, keep_tok + drop_tok:]], dim=1)
        v = torch.cat([v[:, :keep_tok], v[:, keep_tok + drop_tok:]], dim=1)
        if delta_frames != 0 and k.shape[1] > keep_tok:
            vid = k[:, keep_tok:]
            B, S, D = vid.shape
            kc = torch.view_as_complex(vid.to(torch.float64).reshape(B, S, num_heads, head_dim // 2, 2).contiguous())
            kc[..., :n_f] = kc[..., :n_f] * rot.to(kc.device)
            k = torch.cat([k[:, :keep_tok], torch.view_as_real(kc).reshape(B, S, D).to(k.dtype)], dim=1)
        c["k"], c["v"] = k, v
