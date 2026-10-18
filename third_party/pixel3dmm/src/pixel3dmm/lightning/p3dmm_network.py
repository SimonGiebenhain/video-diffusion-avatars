import os.path
from typing import Any

from PIL import Image
import torch, timm, random
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F
from torch.nn.functional import pad
import numpy as np
from torch.cuda.amp import autocast

from pixel3dmm.tools.rsh import rsh_cart_3, rsh_cart_6_2d
from einops.layers.torch import Rearrange
from typing import Optional, Tuple, List

import pytorch_lightning as L
from torchvision import transforms
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d
from torchvision.transforms import Normalize, Resize, Compose

from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.overrides import has_torch_function, has_torch_function_unary, has_torch_function_variadic, \
    handle_torch_function

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
import warnings
import math


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched


class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class DinoWrapper(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """

    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))

        return outputs[:, 1:]

    def freeze(self, is_train: bool = False):
        print(f"======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            data_config = timm.data.resolve_model_data_config(model)
            processor = transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err







def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode for x in torch_dispatch_mode_stack)
    else:
        return False


def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type: Optional[DType],
    other_name: str,
    target_type: DType,
    check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask


def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = F.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    learnable_scale: torch.Tensor = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
            learnable_scale=learnable_scale,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        q = torch.nn.functional.normalize(q, p=2, dim=-1) * math.sqrt(q.shape[-1]) * learnable_scale
        k = torch.nn.functional.normalize(k, p=2, dim=-1) * math.sqrt(q.shape[-1]) * learnable_scale
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class MultiheadAttention_cstm(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiHeadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.learnable_scale = torch.nn.Parameter(torch.ones([], **factory_kwargs), )
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
            and achieve the best performance for MHA.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                learnable_scale=self.learnable_scale)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                learnable_scale=self.learnable_scale)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads,
                                                                                                    -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type


class GroupAttBlock(L.LightningModule):
    def __init__(self, inner_dim: int, input_dim: int,
                 num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.self_attn = MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=inner_dim, vdim=inner_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.self_attn2 = MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=inner_dim, vdim=inner_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)

        self.norm2 = norm_layer(inner_dim)
        self.norm3 = norm_layer(inner_dim)
        self.norm4 = norm_layer(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, facial_components=None):
        # x: [B, C, H, W]
        # cond: [B, L_cond, D_cond]

        B, V, C, H, W = x.shape

        x = x.permute(0, 1, 3, 4, 2).view(B, V * H * W, C)
        if facial_components is not None:
            n_facial_components = facial_components.shape[1]
            x = torch.cat([x, facial_components], dim=1)
        patches = self.norm1(x)
        patches = patches
        # self attention
        patches = patches + self.self_attn(patches, patches, patches, need_weights=False)[0]
        patches = patches + self.mlp(self.norm2(patches))

        patches = self.norm3(patches)
        patches = patches + self.self_attn2(patches, patches, patches, need_weights=False)[0]
        patches = patches + self.mlp2(self.norm4(patches))

        if facial_components is not None:
            facial_components = patches[:, -n_facial_components:, :]
            patches = patches[:, :-n_facial_components, :]
        else:
            facial_components = None

        patches = patches.reshape(B, V, H, W, C).permute(0, 1, 4, 2, 3)

        return patches, facial_components


class Upsampler(L.LightningModule):
    def __init__(self, embedding_dim, window_size):
        super().__init__()

        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.linear_up_1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.pixel_shuffle_1 = nn.PixelShuffle(2)

        self.group = Rearrange('b c (h p1) (w p2) -> b c h w (p1 p2)', p1=window_size, p2=window_size)
        self.ungroup = Rearrange('b h w (p1 p2) c -> b (h p1) (w p2) c', p1=window_size, p2=window_size)

        mlp_ratio = 1
        mlp_drop = 0.0
        self.mlp1 = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(embedding_dim * mlp_ratio), embedding_dim),
            nn.Dropout(mlp_drop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(embedding_dim * mlp_ratio), embedding_dim),
            nn.Dropout(mlp_drop),
        )
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.norm3 = torch.nn.LayerNorm(embedding_dim)
        self.self_attn1_1 = MultiheadAttention(embed_dim=embedding_dim, num_heads=8, kdim=embedding_dim,
                                               vdim=embedding_dim, batch_first=True)
        self.self_attn1_2 = MultiheadAttention(embed_dim=embedding_dim, num_heads=8, kdim=embedding_dim,
                                               vdim=embedding_dim, batch_first=True)

    def forward(self, img_feats):
        b = img_feats.shape[0]

        # image_feats: b x c x h_low x w_low
        img_feats = self.linear_up_1(img_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        img_feats_up = self.pixel_shuffle_1(img_feats)  # b x c x 2*h_low x 2*w_low

        grouped_feats = self.group(img_feats_up)  # b x c x h' x w' x (window_size**2)
        grouped_h = grouped_feats.shape[2]
        grouped_w = grouped_feats.shape[3]
        grouped_feats = grouped_feats.permute(0, 2, 3, 1, 4).reshape(-1, self.embedding_dim,
                                                                     self.window_size ** 2)  # b' x c x win**2
        grouped_feats = grouped_feats.permute(0, 2, 1)
        grouped_feats = self.norm0(grouped_feats)
        grouped_feats = grouped_feats + \
                        self.self_attn1_1(grouped_feats, grouped_feats, grouped_feats, need_weights=False)[
                            0]  # b' x win**2 x c
        grouped_feats = grouped_feats + self.mlp1(self.norm1(grouped_feats))

        # ungroup
        img_feats_up = grouped_feats.reshape(b, grouped_h, grouped_w, self.window_size ** 2,
                                             self.embedding_dim)  # b x h' x w' x win**2 x c
        img_feats_up = self.ungroup(img_feats_up)  # b h w c

        # shift
        img_feats_up = torch.cat(
            [img_feats_up[:, -self.window_size // 2:, :, :], img_feats_up[:, :-self.window_size // 2, :, :]], axis=1)
        img_feats_up = torch.cat(
            [img_feats_up[:, :, -self.window_size // 2:, :], img_feats_up[:, :, :-self.window_size // 2, :]], axis=2)
        img_feats_up = img_feats_up.permute(0, 3, 1, 2)
        grouped_feats = self.group(img_feats_up)  # b x c x h' x w' x (window_size**2)
        grouped_h = grouped_feats.shape[2]
        grouped_w = grouped_feats.shape[3]
        grouped_feats = grouped_feats.permute(0, 2, 3, 1, 4).reshape(-1, self.embedding_dim,
                                                                     self.window_size ** 2)  # b' x c x win**2
        grouped_feats = grouped_feats.permute(0, 2, 1)
        grouped_feats = self.norm2(grouped_feats)
        grouped_feats = grouped_feats + \
                        self.self_attn1_2(grouped_feats, grouped_feats, grouped_feats, need_weights=False)[
                            0]  # b' x win**2 x c
        grouped_feats = grouped_feats + self.mlp2(self.norm3(grouped_feats))

        # ungroup
        img_feats_up = grouped_feats.reshape(b, grouped_h, grouped_w, self.window_size ** 2,
                                             self.embedding_dim)  # b x h' x w' x win**2 x c
        img_feats_up = self.ungroup(img_feats_up)  # b h w c

        # un-shift
        img_feats_up = torch.cat(
            [img_feats_up[:, self.window_size // 2:, :, :], img_feats_up[:, :self.window_size // 2, :, :]], axis=1)
        img_feats_up = torch.cat(
            [img_feats_up[:, :, self.window_size // 2:, :], img_feats_up[:, :, :self.window_size // 2, :]], axis=2)
        img_feats_up = img_feats_up.permute(0, 3, 1, 2)

        return img_feats_up


class VolTransformer(L.LightningModule):
    """
    Transformer with condition and modulation that generates a triplane representation.

    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """

    def __init__(self, embed_dim: int, image_feat_dim: int, n_groups: list,
                 vol_low_res: int, vol_high_res: int, out_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6):
        super().__init__()

        # attributes
        self.vol_low_res = vol_low_res
        self.vol_high_res = vol_high_res
        self.out_dim = out_dim
        self.n_groups = n_groups
        # self.block_size = [vol_low_res//item for item in n_groups]
        self.embed_dim = embed_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.down_proj = torch.nn.Linear(image_feat_dim, embed_dim)

        self.layers = nn.ModuleList([
            GroupAttBlock(
                inner_dim=embed_dim, input_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        # self.deconv = nn.ConvTranspose3d(embed_dim, out_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, image_feats, facial_components=None):
        # image_feats: [B, C, H, W]
        # camera_embeddings: [N, D_mod]

        B, V, C, H, W = image_feats.shape

        image_feats = self.down_proj(image_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

        # self-attention, norm, mlp blocks
        for i, layer in enumerate(self.layers):
            image_feats, facial_components = layer(image_feats, facial_components)

        x = image_feats
        # x = self.norm(torch.einsum('bchw->bhwc',x))
        # x = torch.einsum('bhwc->bchw',x)

        # separate each plane and apply deconv
        # x_up = self.deconv(x)  # [3*N, H', W']
        # x_up = torch.einsum('bchw->bhwc',x_up).contiguous()
        return x, facial_components


def unpatchify(x, batch_size, channels=3, patch_size=16, n_views: int = 1):
    """
    x: (N, L, patch_size**2 *channels)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(batch_size, n_views, h, w, patch_size, patch_size, channels))
    x = torch.einsum('nvhwpqc->nvchpwq', x)
    imgs = x.reshape(shape=(batch_size, n_views, channels, h * patch_size, h * patch_size))
    return imgs


def get_pose_feat(src_exts, tar_ext, src_ixts, W, H):
    """
    src_exts: [B,N,4,4]
    tar_ext: [B,4,4]
    src_ixts: [B,N,3,3]
    """

    B = src_exts.shape[0]
    c2w_ref = src_exts[:, 0].view(B, -1)
    normalize_facto = torch.tensor([W, H]).unsqueeze(0).to(c2w_ref)
    fx_fy = src_ixts[:, 0, [0, 1], [0, 1]] / normalize_facto
    cx_cy = src_ixts[:, 0, [0, 1], [2, 2]] / normalize_facto

    return torch.cat((c2w_ref, fx_fy, fx_fy), dim=-1)


def projection(grid, w2cs, ixts):
    points = grid.reshape(1, -1, 3) @ w2cs[:, :3, :3].permute(0, 2, 1) + w2cs[:, :3, 3][:, None]
    points = points @ ixts.permute(0, 2, 1)
    points_xy = points[..., :2] / points[..., -1:]
    return points_xy, points[..., -1:]


class ModLN(L.LightningModule):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]


class Decoder(L.LightningModule):
    def __init__(self, in_dim, sh_dim, scaling_dim, rotation_dim, opacity_dim, K=1, latent_dim=256, cnn_dim=0):
        super(Decoder, self).__init__()

        self.K = K
        self.sh_dim = sh_dim
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim = rotation_dim
        self.out_dim = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim + cnn_dim
        self.cnn_dim = cnn_dim

        if self.cnn_dim > 0:
            assert sh_dim == 3

        num_layer = 2
        layers_coarse = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
                        [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer - 1) + \
                        [nn.Linear(in_dim, self.out_dim * K)]
        self.mlp_coarse = nn.Sequential(*layers_coarse)

        cond_dim = 8
        self.norm = nn.LayerNorm(in_dim)
        self.cross_att = MultiheadAttention(
            embed_dim=in_dim, num_heads=8, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)
        layers_fine = [nn.Linear(in_dim, 64), nn.ReLU()] + \
                      [nn.Linear(64, self.sh_dim)]
        self.mlp_fine = nn.Sequential(*layers_fine)

        self.init(self.mlp_coarse)
        self.init(self.mlp_fine)

    def init(self, layers):
        # MLP initialization as in mipnerf360
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

    def forward_coarse(self, feats, opacity_shift, scaling_shift):
        parameters = self.mlp_coarse(feats).float()
        parameters = parameters.view(*parameters.shape[:-1], self.K, -1)
        offset, sh, opacity, scaling, rotation = torch.split(
            parameters,
            [3, (self.sh_dim + self.cnn_dim), self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
        )
        opacity = opacity + opacity_shift
        scaling = scaling + scaling_shift
        offset = torch.sigmoid(offset) * 2 - 1.0

        B = opacity.shape[0]
        sh = sh.view(B, -1, self.sh_dim // 3, 3 + self.cnn_dim)
        opacity = opacity.view(B, -1, self.opacity_dim)
        scaling = scaling.view(B, -1, self.scaling_dim)
        rotation = rotation.view(B, -1, self.rotation_dim)
        offset = offset.view(B, -1, 3)

        return offset, sh, scaling, rotation, opacity

    def forward_fine(self, volume_feat, point_feats):
        volume_feat = self.norm(volume_feat.unsqueeze(1))
        x = self.cross_att(volume_feat, point_feats, point_feats, need_weights=False)[0]
        sh = self.mlp_fine(x).float()
        return sh


class Network(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        if not hasattr(cfg.model, 'pred_disentangled'):
            cfg.model.pred_disentangled = False
        if not hasattr(cfg.model, 'use_uv_enc'):
            cfg.model.use_uv_enc = False
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        # modules
        if self.cfg.model.feature_map_type == 'DINO':
            self.img_encoder = DinoWrapper(
                model_name=cfg.model.encoder_backbone,
                is_train=self.cfg.model.finetune_backbone,
            )
            self.feat_map_size = 32
        if self.cfg.model.feature_map_type == 'FaRL':
            self.img_encoder = FaRLWrapperActual(
                model_name=cfg.model.encoder_backbone,
                is_train=self.cfg.model.finetune_backbone,
            )
            self.feat_map_size = 14
        elif self.cfg.model.feature_map_type == 'MICA':
            self.img_encoder = MICA(
                model_name=cfg.model.encoder_backbone,
                # is_train=self.cfg.model.finetune_backbone
            )
            self.forward = self.forward_mica
        elif self.cfg.model.feature_map_type == 'sapiens':
            raise NotImplementedError("feature_map_type='sapiens' (Sapiens backbone) is not included in this copy of "
                                      "pixel3dmm; the released checkpoints use 'DINO'")

        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16 * 2, eps=1e-6)
        self.dir_norm_uv = ModLN(encoder_feat_dim, encoder_feat_dim, eps=1e-6)
        self.uv_enc_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(24, encoder_feat_dim),
        )

        if self.cfg.model.use_pos_enc:
            self.patch_pos_enc = nn.Parameter(
                torch.randn(1, encoder_feat_dim, self.feat_map_size, self.feat_map_size) * (1 / encoder_feat_dim) ** 0.5
            )

        if self.cfg.n_views > 1:
            self.view_embed = nn.Parameter(
                torch.randn(1, self.cfg.n_views, self.cfg.model.view_embed_dim, 1, 1) * (
                        1 / cfg.model.view_embed_dim) ** 0.5  # TODO
            )

            inp_dim_transformer = encoder_feat_dim + cfg.model.view_embed_dim
        else:
            inp_dim_transformer = encoder_feat_dim
        # build volume transformer
        # self.n_groups = cfg.model.n_groups
        embedding_dim = cfg.model.embedding_dim
        self.vol_decoder = VolTransformer(
            embed_dim=embedding_dim, image_feat_dim=inp_dim_transformer,
            vol_low_res=None, vol_high_res=None, out_dim=cfg.model.vol_embedding_out_dim, n_groups=None,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
        )

        self.prediction_dim = 0
        for prediction_type in ['pos_map', 'normals', 'albedo', 'uv_map', 'depth', 'nocs']:
            if prediction_type in self.cfg.model.prediction_type:
                if prediction_type in ['pos_map', 'normals', 'albedo', 'nocs']:
                    self.prediction_dim += 3
                    if prediction_type in ['pos_map', 'normals'] and self.cfg.model.pred_disentangled:
                        self.prediction_dim += 3
                elif prediction_type == 'uv_map':
                    self.prediction_dim += 2
                    if self.cfg.model.pred_disentangled:
                        self.prediction_dim += 2
                elif prediction_type in ['depth', 'depth_si']:
                    self.prediction_dim += 1
        self.pred_disentangled = self.cfg.model.pred_disentangled

        self.t_conv1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)  # 32->64
        self.t_conv2 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)  # 64->128
        self.t_conv3 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)  # 128->256
        # self.t_conv4 = nn.ConvTranspose2d(embedding_dim // 2, self.prediction_dim, 2, stride=2)  # 256->512

        if self.cfg.model.conv_dec:
            remaining_patch_size = 2
        elif self.cfg.model.feature_map_type == 'DINO':
            remaining_patch_size = 16
        else:
            remaining_patch_size = 8

        self.patch_size = remaining_patch_size

        # self.token_2_patch_content = nn.Sequential(
        #        nn.Linear(embedding_dim, embedding_dim),
        #        nn.GELU(),
        #        nn.Linear(embedding_dim, remaining_patch_size**2*self.prediction_dim),
        #        #nn.Linear(embedding_dim, 16*16*self.prediction_dim),
        # )
        self.token_2_patch_content = nn.Linear(embedding_dim, remaining_patch_size ** 2 * self.prediction_dim)

        if self.cfg.model.pred_conf:
            self.t_conv3_conf = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)
            self.token_2_patch_conf = nn.Linear(embedding_dim, remaining_patch_size ** 2 * 1)

        self.n_facial_components = cfg.model.n_facial_components if hasattr(cfg.model, 'n_facial_components') else 0
        if self.n_facial_components > 0:
            self.facial_components = nn.Parameter(torch.zeros([self.n_facial_components,
                                                               embedding_dim]))  # torch.nn.Embedding(self.n_facial_components, embedding_dim)
            # nn.init.trunc_normal_(self.facial_components, std=0.02)
            # with torch.no_grad():
            #    self.facial_components.weight = nn.Parameter(torch.zeros_like(self.facial_components.weight))

            self.head_shape = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                            nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                            nn.Linear(embedding_dim, self.cfg.model.flame_shape_dim))
            self.head_expr = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                           nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),  # TODO
                                           nn.Linear(embedding_dim, self.cfg.model.flame_expr_dim))
            # self.head_jaw = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
            #                                nn.Linear(embedding_dim, 6))
            self.head_focal_length = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                                   nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                                   nn.Linear(embedding_dim, 2))
            self.head_principal_point = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                                      nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                                      nn.Linear(embedding_dim, 2))
            self.head_cam_pos = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                              nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                              nn.Linear(embedding_dim, 3))
            self.head_cam_rot = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                              nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(),
                                              nn.Linear(embedding_dim, 6))


        # 32x32-->64x64
        # self.up1 = Upsampler(embedding_dim, 8)
        # self.up2 = Upsampler(embedding_dim, 8)
        ##self.up3 = Upsampler(embedding_dim, 8)
        ##self.up4 = Upsampler(embedding_dim, 8)
        # self.lin_up = torch.nn.Linear(embedding_dim, self.prediction_dim*4*4)
        ##self.lin_out = torch.nn.Linear(embedding_dim, self.prediction_dim)

        # self.feat_vol_reso = cfg.model.vol_feat_reso
        # self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))

        # grouping configuration
        # self.n_offset_groups = cfg.model.n_offset_groups
        # self.register_buffer("group_centers", self.build_dense_grid(self.grid_reso*2))
        # self.group_centers = self.group_centers.reshape(1,-1,3)

        # 2DGS model
        # self.sh_dim = (cfg.model.sh_degree+1)**2*3
        # self.scaling_dim, self.rotation_dim = 2, 4
        # self.opacity_dim = 1
        # self.out_dim = self.sh_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim

        # self.K = cfg.model.K
        # vol_embedding_out_dim = cfg.model.vol_embedding_out_dim
        # self.decoder = Decoder(vol_embedding_out_dim, self.sh_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, self.K,
        #                       cnn_dim=cfg.model.cnn_dim)
        # self.gs_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1)

        # parameters initialization
        # self.opacity_shift = -2.1792
        # self.voxel_size = 2.0/(self.grid_reso*2)
        # self.scaling_shift = np.log(0.5*self.voxel_size/3.0)

        # self.has_cnn = cfg.model.cnn_dim > 0
        # assert cfg.model.cnn_dim <= 13
        # if self.has_cnn:
        #    self.cnn = Upsampler()
        # self.cnn_dim = cfg.model.cnn_dim

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'), dim=-1)
        grid = (grid + 0.5) / reso * 2 - 1
        return grid.reshape(reso, reso, reso, 3) * self.scene_size

    def add_pos_enc_patches(self, src_inps, img_feats, n_views_sel, batch):

        h, w = src_inps.shape[-2:]
        # src_ixts = batch['tar_ixt'][:,:n_views_sel].reshape(-1,3,3)
        # src_w2cs = batch['tar_w2c'][:,:n_views_sel].reshape(-1,4,4)

        # img_wh = torch.tensor([w,h], device=self.device)
        # point_img,_ = projection(self.volume_grid, src_w2cs, src_ixts)
        # point_img = (point_img+ 0.5)/img_wh*2 - 1.0

        # viewing direction
        rays = batch['tar_rays_down'][:, :n_views_sel]
        feats_dir = self.ray_to_plucker(rays).reshape(-1, *rays.shape[2:])
        feats_dir = torch.cat((rsh_cart_3(feats_dir[..., :3]), rsh_cart_3(feats_dir[..., 3:6])), dim=-1)

        # query features
        img_feats = torch.einsum('bchw->bhwc', img_feats)
        # print('img_feats.shape:', img_feats.shape)
        # print('feats_dir.shape:', feats_dir.shape)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum('bhwc->bchw', img_feats)

        # n_channel = img_feats.shape[1]
        # feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        ## img features
        # feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)
        c, h, w = img_feats.shape[1:]
        img_feats = img_feats.reshape(-1, n_views_sel, c, h, w)
        return img_feats

    def add_uv_enc_patches(self, src_inps, img_feats, n_views_sel, batch):

        h, w = src_inps.shape[-2:]

        # viewing direction
        rays = batch['tar_uvs_down'][:, :n_views_sel]
        feats_dir = rsh_cart_6_2d(rays)

        # query features
        img_feats = torch.einsum('bchw->bhwc', img_feats)

        # print('img_feats.shape:', img_feats.shape)
        # print('feats_dir.shape:', feats_dir.shape)
        feats_dir = self.uv_enc_mlp(feats_dir)
        img_feats = img_feats.reshape(feats_dir.shape[0], feats_dir.shape[1], img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        img_feats = self.dir_norm_uv(img_feats, feats_dir)
        img_feats = torch.einsum('bvhwc->bvchw', img_feats)

        # n_channel = img_feats.shape[1]
        # feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        ## img features
        # feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)
        # c, h, w = img_feats.shape[1:]
        # img_feats = img_feats.reshape(-1, n_views_sel, c, h, w)
        return img_feats

    def add_pixel_pred_patches(self, src_inps, img_feats, n_views_sel, batch):

        rays = batch['tar_ns_down'][:, :n_views_sel]
        rays = rays.reshape(-1, *rays.shape[2:])
        uvs = batch['tar_uvs_down'][:, :n_views_sel]
        uvs = uvs.reshape(-1, *uvs.shape[2:])
        feats_dir = torch.cat((
            rsh_cart_3(rays[..., :3]),
            rsh_cart_3(torch.cat([uvs, torch.zeros_like(uvs[..., -1:])], dim=-1))
        ), dim=-1)

        # query features
        img_feats = torch.einsum('bchw->bhwc', img_feats)
        # print('img_feats.shape:', img_feats.shape)
        # print('feats_dir.shape:', feats_dir.shape)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum('bhwc->bchw', img_feats)

        # n_channel = img_feats.shape[1]
        # feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        ## img features
        # feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)
        c, h, w = img_feats.shape[1:]
        img_feats = img_feats.reshape(-1, n_views_sel, c, h, w)
        return img_feats

    def _check_mask(self, mask):
        ratio = torch.sum(mask) / np.prod(mask.shape)
        if ratio < 1e-3:
            mask = mask + torch.rand(mask.shape, device=self.device) > 0.8
        elif ratio > 0.5 and self.training:
            # avoid OMM
            mask = mask * torch.rand(mask.shape, device=self.device) > 0.5
        return mask

    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):

        points = points[mask]
        n_points = points.shape[0]

        h, w = img_ref.shape[-2:]
        src_ixts = batch['tar_ixt'][idx, :n_views_sel].reshape(-1, 3, 3)
        src_w2cs = batch['tar_w2c'][idx, :n_views_sel].reshape(-1, 4, 4)

        img_wh = torch.tensor([w, h], device=self.device)
        point_xy, point_z = projection(points, src_w2cs, src_ixts)
        point_xy = (point_xy + 0.5) / img_wh * 2 - 1.0

        imgs_coarse = torch.cat((renderings['image'], renderings['acc_map'].unsqueeze(-1), renderings['depth']), dim=-1)
        imgs_coarse = torch.cat((img_ref, torch.einsum('bhwc->bchw', imgs_coarse)), dim=1)
        feats_coarse = F.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False).view(n_views_sel, -1,
                                                                                                   n_points).to(
            imgs_coarse)

        z_diff = (feats_coarse[:, -1:] - point_z.view(n_views_sel, -1, n_points)).abs()

        point_feats = torch.cat((feats_coarse[:, :-1], z_diff), dim=1)  # [...,_mask]

        return point_feats, mask

    def ray_to_plucker(self, rays):
        origin, direction = rays[..., :3], rays[..., 3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)

        # Calculate the moment vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)

        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment), dim=-1)

    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5 * self.scene_size / self.n_offset_groups
        centers = self.group_centers.unsqueeze(-2).expand(B, -1, K, -1).reshape(offset.shape) + offset * half_cell_size
        return centers

    def forward_new(self, batch, return_feature_map: bool = False, input_name='tar_rgb'):
        og_tar_rgb = batch['tar_rgb']
        batch['tar_rgb'] = batch[input_name]
        B, N, H, W, C = batch['tar_rgb'].shape
        # if self.training:
        #    n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        # else:
        n_views_sel = N  # self.cfg.n_views

        _inps = batch['tar_rgb'][:, :n_views_sel].reshape(B * n_views_sel, H, W, C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        # image encoder
        if self.cfg.model.feature_map_type == 'sapiens':
            if self.cfg.model.finetune_backbone:
                _inps = self.bicubic_up(_inps)
                img_feats = self.img_encoder(_inps)
            else:
                with torch.no_grad():
                    _inps = self.bicubic_up(_inps)
                    img_feats = self.img_encoder(_inps)

        elif self.cfg.model.feature_map_type == 'DINO':
            if self.cfg.model.finetune_backbone:
                img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            else:
                with torch.no_grad():
                    img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
            img_feats = img_feats.reshape(*img_feats.shape[:2], H // token_size, W // token_size)

        elif self.cfg.model.feature_map_type == 'FaRL':
            if self.cfg.model.finetune_backbone:
                img_feats = self.img_encoder(_inps, facial_components=self.facial_components)
            else:
                with torch.no_grad():
                    img_feats = self.img_encoder(_inps, facial_components=self.facial_components)
            facial_components = img_feats[:, -6:, :]

        out_dict = {}
        flame_shape = self.head_shape(facial_components[:, 0, :])
        flame_expr = self.head_expr(facial_components[:, 1, :])
        # flame_jaw = self.head_jaw(facial_components[:, 2, :])
        base_rot = torch.zeros([B, 6], device=flame_shape.device)
        base_rot[:, 0] = -1
        base_rot[:, 5] = 1
        flame_focal_length = self.head_focal_length(facial_components[:, 3, :])
        flame_principal_point = self.head_principal_point(facial_components[:, 2, :])
        cam_pos = self.head_cam_pos(facial_components[:, 4, :])
        cam_rot = self.head_cam_rot(facial_components[:, 5, :])
        out_dict['shape'] = flame_shape  # * self.std_id + self.mean_id
        out_dict['expr'] = flame_expr  # * self.std_ex + self.mean_ex
        # out_dict['jaw'] = base_rot + flame_jaw
        out_dict['focal_length'] = flame_focal_length
        out_dict['principal_point'] = flame_principal_point
        out_dict['cam_c2w_pos'] = cam_pos
        out_dict['cam_c2w_rot'] = rotation_6d_to_matrix(base_rot + cam_rot)

        batch['tar_rgb'] = og_tar_rgb

        # for k in out_dict.keys():
        # print(k, out_dict[k].shape)
        return out_dict, None

    def forward_hybrid(self, batch, return_feature_map: bool = False):

        B, N, H, W, C = batch['tar_rgb'].shape
        # if self.training:
        #    n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        # else:
        n_views_sel = N  # self.cfg.n_views

        _inps = batch['tar_rgb'][:, :n_views_sel].reshape(B * n_views_sel, H, W, C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        # image encoder
        if self.cfg.model.feature_map_type == 'sapiens':
            if self.cfg.model.finetune_backbone:
                _inps = self.bicubic_up(_inps)
                img_feats = self.img_encoder(_inps)
            else:
                with torch.no_grad():
                    _inps = self.bicubic_up(_inps)
                    img_feats = self.img_encoder(_inps)

        elif self.cfg.model.feature_map_type == 'DINO':
            if self.cfg.model.finetune_backbone:
                img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            else:
                with torch.no_grad():
                    img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
            img_feats = img_feats.reshape(*img_feats.shape[:2], H // token_size, W // token_size)

        elif self.cfg.model.feature_map_type == 'FaRL':
            if self.cfg.model.finetune_backbone:
                img_feats = self.img_encoder(_inps, facial_components=self.facial_components)
            else:
                with torch.no_grad():
                    img_feats = self.img_encoder(_inps, facial_components=self.facial_components)
            facial_components = img_feats[:, -6:, :]
            img_feats = img_feats[:, :-6, :]
            img_feats = img_feats.permute(0, 2, 1)
            token_size = int(np.sqrt(224 * 224 / img_feats.shape[-1]))
            img_feats = img_feats.reshape(*img_feats.shape[:2], 224 // token_size, 224 // token_size)

        if self.cfg.model.use_pos_enc:
            img_feats = img_feats + self.patch_pos_enc

        # print(img_feats.shape)
        if hasattr(self.cfg.model, 'prior_input') and self.cfg.model.prior_input:  # self.cfg.model.use_pixel_preds:
            img_feats = self.add_pixel_pred_patches(_inps, img_feats, n_views_sel, batch).squeeze(
                1)  # B n_views_sel C H W
        # print(img_feats.shape)
        # exit()

        if self.cfg.model.use_plucker:
            img_feats = self.add_pos_enc_patches(_inps, img_feats, n_views_sel, batch)  # B n_views_sel C H W
        else:
            img_feats = img_feats.reshape(B, N, img_feats.shape[1], img_feats.shape[2], img_feats.shape[3])
        if self.cfg.n_views > 1:
            img_feats = torch.cat((img_feats,
                                   self.view_embed[:, :n_views_sel].repeat(B, 1, 1, img_feats.shape[-2],
                                                                           img_feats.shape[-1])), dim=2)

        # decoding
        img_feats, facial_components = self.vol_decoder(img_feats, facial_components=facial_components)  # b c h w

        out_dict = {}

        if self.n_facial_components == 0:
            img_feats = img_feats.reshape(-1, img_feats.shape[2], img_feats.shape[3], img_feats.shape[4])

            if False:
                img_feats = self.up1(img_feats)
                img_feats = self.up2(img_feats)
                img_feats = img_feats.permute(0, 2, 3, 1)
                img_feats = img_feats.reshape(img_feats.shape[0], -1, img_feats.shape[-1])  # b l c
                img_feats = self.lin_up(img_feats)  # b l 16*16*3
                # #img_feats = self.up3(img_feats)
                # img_feats = self.up4(img_feats)
                img = unpatchify(img_feats, channels=self.prediction_dim, patch_size=4)  # b 3 h_full w_full
                # img = self.lin_out(img_feats)
            if self.cfg.model.conv_dec:
                if self.cfg.model.feature_map_type == 'DINO':
                    img_feats = F.gelu(self.t_conv1(img_feats, output_size=(64, 64)))
                img_feats = F.gelu(self.t_conv2(img_feats, output_size=(128, 128)))
                if self.cfg.model.pred_conf:
                    conf_feats = F.gelu(self.t_conv3_conf(img_feats, output_size=(256, 256)))
                img_feats = F.gelu(self.t_conv3(img_feats, output_size=(256, 256)))

                # img = self.t_conv4(img_feats, output_size=(512, 512)).squeeze()

            # self.cfg.model.prediction_type in ['pos_map', 'uv_map', 'normals', 'depth', 'depth_si', 'albedo']:
            img_feats = img_feats.permute(0, 2, 3, 1)
            img_feats = img_feats.reshape(img_feats.shape[0], -1, img_feats.shape[-1])  # b l c
            img_feats = self.token_2_patch_content(img_feats)  # b l 16*16*3
            img = unpatchify(img_feats, batch_size=B, channels=self.prediction_dim, patch_size=self.patch_size,
                             n_views=n_views_sel)  # b 3 h_full w_full
            if self.cfg.model.pred_conf:
                conf_feats = conf_feats.permute(0, 2, 3, 1)
                conf_feats = conf_feats.reshape(img_feats.shape[0], -1, conf_feats.shape[-1])  # b l c
                conf_feats = self.token_2_patch_conf(conf_feats)  # b l 16*16*3
                conf = unpatchify(conf_feats, batch_size=B, channels=1, patch_size=self.patch_size,
                                  n_views=n_views_sel)  # b 3 h_full w_full
            else:
                conf = None

            cur_dim = 0
            if 'pos_map' in self.cfg.model.prediction_type:
                out_dict['pos_map'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
                if self.pred_disentangled:
                    out_dict['pos_map_can'] = img[:, :, cur_dim:cur_dim + 3, ...]
                    cur_dim += 3
            if 'uv_map' in self.cfg.model.prediction_type:
                out_dict['uv_map'] = img[:, :, cur_dim:cur_dim + 2, ...]
                cur_dim += 2
                if self.pred_disentangled:
                    out_dict['disps'] = img[:, :, cur_dim:cur_dim + 2, ...]
                    cur_dim += 2
            if 'normals' in self.cfg.model.prediction_type:
                out_dict['normals'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
                if self.pred_disentangled:
                    out_dict['normals_can'] = img[:, :, cur_dim:cur_dim + 3, ...]
                    cur_dim += 3
            if 'albedo' in self.cfg.model.prediction_type:
                out_dict['albedo'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3

            if 'nocs' in self.cfg.model.prediction_type:
                out_dict['nocs'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
        else:
            conf = None

        if facial_components is not None:
            flame_shape = self.head_shape(facial_components[:, 0, :])
            flame_expr = self.head_expr(facial_components[:, 1, :])
            # flame_jaw = self.head_jaw(facial_components[:, 2, :])
            base_rot = torch.zeros([B, 6], device=flame_shape.device)
            base_rot[:, 0] = -1
            base_rot[:, 5] = 1
            flame_focal_length = self.head_focal_length(facial_components[:, 3, :])
            flame_principal_point = self.head_principal_point(facial_components[:, 2, :])
            cam_pos = self.head_cam_pos(facial_components[:, 4, :])
            cam_rot = self.head_cam_rot(facial_components[:, 5, :])
            out_dict['shape'] = flame_shape  # * self.std_id + self.mean_id
            out_dict['expr'] = flame_expr  # * self.std_ex + self.mean_ex
            # out_dict['jaw'] = base_rot + flame_jaw
            out_dict['focal_length'] = flame_focal_length
            out_dict['principal_point'] = flame_principal_point
            out_dict['cam_c2w_pos'] = cam_pos
            out_dict['cam_c2w_rot'] = rotation_6d_to_matrix(base_rot + cam_rot)

        # for k in out_dict.keys():
        # print(k, out_dict[k].shape)
        return out_dict, conf

    def forward(self, batch, return_feature_map: bool = False, input_name='tar_rgb'):
        og_tar_rgb = batch['tar_rgb']
        # batch['tar_rgb'] = batch[input_name]
        B, N, H, W, C = batch['tar_rgb'].shape
        # if self.training:
        #    n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        # else:
        n_views_sel = N  # self.cfg.n_views

        # if self.n_facial_components > 0:
        #    facial_components = self.facial_components.unsqueeze(0).repeat(B, 1, 1)
        # else:
        #    facial_components = None
        if self.n_facial_components == 0:
            facial_components = None
        _inps = batch['tar_rgb'][:, :n_views_sel].reshape(B * n_views_sel, H, W, C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        # image encoder
        if self.cfg.model.feature_map_type == 'sapiens':
            if self.cfg.model.finetune_backbone:
                _inps = self.bicubic_up(_inps)
                img_feats = self.img_encoder(_inps)
            else:
                with torch.no_grad():
                    _inps = self.bicubic_up(_inps)
                    img_feats = self.img_encoder(_inps)

        elif self.cfg.model.feature_map_type == 'DINO':
            if self.cfg.model.finetune_backbone:
                img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            else:
                with torch.no_grad():
                    img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
            img_feats = img_feats.reshape(*img_feats.shape[:2], H // token_size, W // token_size)
            if self.n_facial_components <= 0:
                facial_components = None
        elif self.cfg.model.feature_map_type == 'FaRL':
            if self.cfg.model.finetune_backbone:
                img_feats, facial_components = self.img_encoder(_inps, facial_components=self.facial_components)
            else:
                with torch.no_grad():
                    img_feats, facial_components = self.img_encoder(_inps, facial_components=self.facial_components)
            # facial_components = img_feats[:, -6:, :]
            # img_feats = img_feats[:, 1:-6, :]

            # img_feats = img_feats.permute(0, 2, 1)
            token_size = int(np.sqrt(224 * 224 / img_feats.shape[-1]))
            # img_feats = img_feats.reshape(*img_feats.shape[:2], 224 // token_size, 224 // token_size)

        if self.cfg.model.use_pos_enc:
            img_feats = img_feats + self.patch_pos_enc

        # print(img_feats.shape)
        if hasattr(self.cfg.model, 'prior_input') and self.cfg.model.prior_input:  # self.cfg.model.use_pixel_preds:
            img_feats = self.add_pixel_pred_patches(_inps, img_feats, n_views_sel, batch).squeeze(
                1)  # B n_views_sel C H W
        # print(img_feats.shape)
        # exit()
        if self.cfg.model.use_uv_enc:
            img_feats = self.add_uv_enc_patches(_inps, img_feats, n_views_sel, batch)  # B n_views_sel C H W
        elif self.cfg.model.use_plucker:
            img_feats = self.add_pos_enc_patches(_inps, img_feats, n_views_sel, batch)  # B n_views_sel C H W
        else:
            img_feats = img_feats.reshape(B, N, img_feats.shape[1], img_feats.shape[2], img_feats.shape[3])
        if self.cfg.n_views > 1:
            img_feats = torch.cat((img_feats,
                                   self.view_embed[:, :n_views_sel].repeat(B, 1, 1, img_feats.shape[-2],
                                                                           img_feats.shape[-1])), dim=2)

        # decoding
        img_feats, facial_components = self.vol_decoder(img_feats, facial_components=facial_components)  # b c h w

        out_dict = {}

        if self.n_facial_components == 0:
            img_feats = img_feats.reshape(-1, img_feats.shape[2], img_feats.shape[3], img_feats.shape[4])

            if False:
                img_feats = self.up1(img_feats)
                img_feats = self.up2(img_feats)
                img_feats = img_feats.permute(0, 2, 3, 1)
                img_feats = img_feats.reshape(img_feats.shape[0], -1, img_feats.shape[-1])  # b l c
                img_feats = self.lin_up(img_feats)  # b l 16*16*3
                # #img_feats = self.up3(img_feats)
                # img_feats = self.up4(img_feats)
                img = unpatchify(img_feats, channels=self.prediction_dim, patch_size=4)  # b 3 h_full w_full
                # img = self.lin_out(img_feats)
            if self.cfg.model.conv_dec:
                if self.cfg.model.feature_map_type == 'DINO':
                    img_feats = F.gelu(self.t_conv1(img_feats, output_size=(64, 64)))
                img_feats = F.gelu(self.t_conv2(img_feats, output_size=(128, 128)))
                if self.cfg.model.pred_conf:
                    conf_feats = F.gelu(self.t_conv3_conf(img_feats, output_size=(256, 256)))
                img_feats = F.gelu(self.t_conv3(img_feats, output_size=(256, 256)))

                # img = self.t_conv4(img_feats, output_size=(512, 512)).squeeze()

            # self.cfg.model.prediction_type in ['pos_map', 'uv_map', 'normals', 'depth', 'depth_si', 'albedo']:
            img_feats = img_feats.permute(0, 2, 3, 1)
            img_feats = img_feats.reshape(img_feats.shape[0], -1, img_feats.shape[-1])  # b l c
            img_feats = self.token_2_patch_content(img_feats)  # b l 16*16*3
            img = unpatchify(img_feats, batch_size=B, channels=self.prediction_dim, patch_size=self.patch_size,
                             n_views=n_views_sel)  # b 3 h_full w_full
            if self.cfg.model.pred_conf:
                conf_feats = conf_feats.permute(0, 2, 3, 1)
                conf_feats = conf_feats.reshape(img_feats.shape[0], -1, conf_feats.shape[-1])  # b l c
                conf_feats = self.token_2_patch_conf(conf_feats)  # b l 16*16*3
                conf = unpatchify(conf_feats, batch_size=B, channels=1, patch_size=self.patch_size,
                                  n_views=n_views_sel)  # b 3 h_full w_full
            else:
                conf = None

            cur_dim = 0
            if 'pos_map' in self.cfg.model.prediction_type:
                out_dict['pos_map'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
                if self.pred_disentangled:
                    out_dict['pos_map_can'] = img[:, :, cur_dim:cur_dim + 3, ...]
                    cur_dim += 3
            if 'uv_map' in self.cfg.model.prediction_type:
                out_dict['uv_map'] = img[:, :, cur_dim:cur_dim + 2, ...]
                cur_dim += 2
                if self.pred_disentangled:
                    out_dict['disps'] = img[:, :, cur_dim:cur_dim + 2, ...]
                    cur_dim += 2
            if 'normals' in self.cfg.model.prediction_type:
                out_dict['normals'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
                if self.pred_disentangled:
                    out_dict['normals_can'] = img[:, :, cur_dim:cur_dim + 3, ...]
                    cur_dim += 3
            if 'albedo' in self.cfg.model.prediction_type:
                out_dict['albedo'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3

            if 'nocs' in self.cfg.model.prediction_type:
                out_dict['nocs'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
        else:
            conf = None

        if facial_components is not None:
            flame_shape = self.head_shape(facial_components[:, 0, :])
            flame_expr = self.head_expr(facial_components[:, 1, :])
            # flame_jaw = self.head_jaw(facial_components[:, 2, :])
            base_rot = torch.zeros([B, 6], device=flame_shape.device)
            base_rot[:, 0] = -1
            base_rot[:, 5] = 1
            flame_focal_length = self.head_focal_length(facial_components[:, 3, :])
            flame_principal_point = self.head_principal_point(facial_components[:, 2, :])
            cam_pos = self.head_cam_pos(facial_components[:, 4, :])
            cam_rot = self.head_cam_rot(facial_components[:, 5, :])
            out_dict['shape'] = flame_shape  # * self.std_id + self.mean_id
            out_dict['expr'] = flame_expr  # * self.std_ex + self.mean_ex
            # out_dict['jaw'] = base_rot + flame_jaw
            out_dict['focal_length'] = flame_focal_length
            out_dict['principal_point'] = flame_principal_point
            out_dict['cam_c2w_pos'] = cam_pos
            out_dict['cam_c2w_rot'] = rotation_6d_to_matrix(base_rot + cam_rot)

        batch['tar_rgb'] = og_tar_rgb

        # for k in out_dict.keys():
        # print(k, out_dict[k].shape)
        return out_dict, conf

    def forward_mica(self, batch, return_feature_map: bool = False, input_name='tar_rgb'):
        _, flame_shape = self.img_encoder(batch['rgb_arcface'])
        out_dict = {}
        conf = None
        out_dict['shape'] = flame_shape
        out_dict['expr'] = torch.zeros_like(flame_shape[..., :100])
        out_dict['focal_length'] = torch.zeros_like(flame_shape[..., :2])
        out_dict['principal_point'] = torch.zeros_like(flame_shape[..., :2])
        out_dict['cam_c2w_pos'] = torch.zeros_like(flame_shape[..., :3])
        out_dict['cam_c2w_rot'] = torch.zeros_like(rotation_6d_to_matrix(flame_shape[..., :6]))

        return out_dict, conf


class Network_cnn(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network_cnn, self).__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        # modules
        # if self.cfg.model.feature_map_type == 'DINO':
        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=self.cfg.model.finetune_backbone,
        )

        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16 * 2, eps=1e-6)

        # build volume transformer
        # self.n_groups = cfg.model.n_groups
        embedding_dim = cfg.model.embedding_dim * 10

        self.embed_mlp = nn.Linear(encoder_feat_dim, embedding_dim)
        self.activation = nn.ReLU()

        self.feature_map_type = self.cfg.model.feature_map_type

        if self.feature_map_type == 'scratch':
            self.cstm_enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)  # 512
            self.cstm_enc_pool = nn.MaxPool2d(2, 2)
            self.cstm_enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)  # 256
            self.cstm_enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # 128
            self.cstm_enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # 64
            self.cstm_enc_conv5 = nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1)  # 32

        if self.feature_map_type == 'arcface':
            raise NotImplementedError("feature_map_type='arcface' (ArcFace backbone) is not included in this copy of "
                                      "pixel3dmm; the released checkpoints use 'DINO'")
        if self.feature_map_type == 'mica':
            self.mica = construct_mica()
            if not self.cfg.model.finetune_backbone:
                # freeze arc face for now
                for name, param in self.mica.named_parameters():
                    param.requires_grad = False

        self.conv1 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        z_dim = 512

        self.conv3 = nn.Conv2d(embedding_dim, z_dim, kernel_size=3, stride=1)

        # self.vol_decoder = VolTransformer(
        #    embed_dim=embedding_dim, image_feat_dim=encoder_feat_dim,  # +cfg.model.view_embed_dim,
        #    vol_low_res=None, vol_high_res=None, out_dim=cfg.model.vol_embedding_out_dim, n_groups=None,
        #    num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
        # )

        self.vertex_encoder = nn.Sequential(
            nn.Linear(5023 * 3, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, z_dim), nn.LeakyReLU(0.2),
        )

        map_hidden_dim = 128
        self.network = nn.ModuleList(
            [nn.Linear(z_dim, map_hidden_dim)] +
            [nn.Linear(map_hidden_dim, map_hidden_dim) for i in range(3)]
        )

        self.output = nn.Linear(map_hidden_dim, 101)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.output.weight *= 0.25

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'), dim=-1)
        grid = (grid + 0.5) / reso * 2 - 1
        return grid.reshape(reso, reso, reso, 3) * self.scene_size

    def add_pos_enc_patches(self, src_inps, img_feats, n_views_sel, batch):

        h, w = src_inps.shape[-2:]
        # src_ixts = batch['tar_ixt'][:,:n_views_sel].reshape(-1,3,3)
        # src_w2cs = batch['tar_w2c'][:,:n_views_sel].reshape(-1,4,4)

        # img_wh = torch.tensor([w,h], device=self.device)
        # point_img,_ = projection(self.volume_grid, src_w2cs, src_ixts)
        # point_img = (point_img+ 0.5)/img_wh*2 - 1.0

        # viewing direction
        rays = batch['tar_rays_down'][:, :n_views_sel]
        feats_dir = self.ray_to_plucker(rays).reshape(-1, *rays.shape[2:])
        feats_dir = torch.cat((rsh_cart_3(feats_dir[..., :3]), rsh_cart_3(feats_dir[..., 3:6])), dim=-1)

        # query features
        img_feats = torch.einsum('bchw->bhwc', img_feats)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum('bhwc->bchw', img_feats)

        # n_channel = img_feats.shape[1]
        # feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        ## img features
        # feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)

        return img_feats

    def _check_mask(self, mask):
        ratio = torch.sum(mask) / np.prod(mask.shape)
        if ratio < 1e-3:
            mask = mask + torch.rand(mask.shape, device=self.device) > 0.8
        elif ratio > 0.5 and self.training:
            # avoid OMM
            mask = mask * torch.rand(mask.shape, device=self.device) > 0.5
        return mask

    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):

        points = points[mask]
        n_points = points.shape[0]

        h, w = img_ref.shape[-2:]
        src_ixts = batch['tar_ixt'][idx, :n_views_sel].reshape(-1, 3, 3)
        src_w2cs = batch['tar_w2c'][idx, :n_views_sel].reshape(-1, 4, 4)

        img_wh = torch.tensor([w, h], device=self.device)
        point_xy, point_z = projection(points, src_w2cs, src_ixts)
        point_xy = (point_xy + 0.5) / img_wh * 2 - 1.0

        imgs_coarse = torch.cat((renderings['image'], renderings['acc_map'].unsqueeze(-1), renderings['depth']), dim=-1)
        imgs_coarse = torch.cat((img_ref, torch.einsum('bhwc->bchw', imgs_coarse)), dim=1)
        feats_coarse = F.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False).view(n_views_sel, -1,
                                                                                                   n_points).to(
            imgs_coarse)

        z_diff = (feats_coarse[:, -1:] - point_z.view(n_views_sel, -1, n_points)).abs()

        point_feats = torch.cat((feats_coarse[:, :-1], z_diff), dim=1)  # [...,_mask]

        return point_feats, mask

    def ray_to_plucker(self, rays):
        origin, direction = rays[..., :3], rays[..., 3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)

        # Calculate the moment vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)

        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment), dim=-1)

    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5 * self.scene_size / self.n_offset_groups
        centers = self.group_centers.unsqueeze(-2).expand(B, -1, K, -1).reshape(offset.shape) + offset * half_cell_size
        return centers

    def forward(self, batch, return_feature_map: bool = False):

        B, N, H, W, C = batch['tar_rgb'].shape
        # if self.training:
        #    n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        # else:
        n_views_sel = 1  # self.cfg.n_views

        _inps = batch['tar_rgb'][:, :n_views_sel].reshape(B * n_views_sel, H, W, C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        # image encoder
        if self.feature_map_type == 'DINO':
            img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
            img_feats = img_feats.reshape(*img_feats.shape[:2], H // token_size, W // token_size)
            img_feats = img_feats.permute(0, 2, 3, 1)
            img_feats = self.activation(self.embed_mlp(img_feats)).permute(0, 3, 1, 2)  # b c h w
        elif self.feature_map_type == 'scratch':
            img_feats = self.cstm_enc_pool(F.leaky_relu(self.cstm_enc_conv1(_inps), negative_slope=0.2))
            img_feats = self.cstm_enc_pool(F.leaky_relu(self.cstm_enc_conv2(img_feats), negative_slope=0.2))
            img_feats = F.leaky_relu(self.cstm_enc_conv3(img_feats), negative_slope=0.2)
            img_feats = self.cstm_enc_pool(F.leaky_relu(self.cstm_enc_conv4(img_feats), negative_slope=0.2))
            img_feats = F.leaky_relu(self.cstm_enc_conv5(img_feats), negative_slope=0.2)
        elif self.feature_map_type == 'arcface':
            img_feats = F.normalize(self.arcface(_inps))
            x = img_feats
        elif self.feature_map_type == 'mica':
            flame_code_pred = self.mica(_inps)[:, :101]  # dirty hack to simulate scale at index 100
            feat_map = None
            if return_feature_map:
                return flame_code_pred, feat_map
            else:
                return flame_code_pred

        if return_feature_map:
            feat_map = img_feats.detach().clone()

        ## build 3D volume
        # TODO add plucker coordinates back
        # img_feats = self.add_pos_enc_patches(_inps, img_feats, n_views_sel, batch) # B n_views_sel C H W

        # decoding
        # img_feats = self.vol_decoder(img_feats)  # b c h w

        if not self.feature_map_type == 'arcface':
            x = img_feats
            x = self.pool1(F.leaky_relu(self.conv1(x), negative_slope=0.2))  # 16x16
            x = self.pool2(F.leaky_relu(self.conv2(x), negative_slope=0.2))  # 8x8
            img_feats = F.leaky_relu(self.conv3(x), negative_slope=0.2)

            # flame_pred
            img_feats = img_feats.reshape(img_feats.shape[0], img_feats.shape[1], -1)  # B C H*W
            x = img_feats.max(-1)[0]  # b c

        for i_layer, layer in enumerate(self.network):
            # if i_layer == 0:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            # else:
            #    x = F.leaky_relu(layer(torch.cat([x, enc], dim=-1)), negative_slope=0.2)

        flame_code_pred = self.output(x)

        if return_feature_map:
            return flame_code_pred, feat_map
        return flame_code_pred


class NetworkSanity(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        # modules
        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=cfg.model.finetune_backbone,
        )

        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16 * 2, eps=1e-6)

        # build volume transformer
        # self.n_groups = cfg.model.n_groups
        embedding_dim = cfg.model.embedding_dim * 10

        self.embed_mlp = nn.Linear(encoder_feat_dim, embedding_dim)
        self.activation = nn.ReLU()

        # self.pred_head = torch.nn.Linear(embedding_dim, cfg.model.flame_dim)
        mlp_ratio = 2
        self.pred_head = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim * mlp_ratio), int(embedding_dim * mlp_ratio)),
            nn.ReLU(),
            # nn.Dropout(mlp_drop),
            nn.Linear(int(embedding_dim * mlp_ratio), cfg.model.flame_dim),
            # nn.Dropout(mlp_drop),
        )
        z_dim = 256

        self.vertex_encoder = nn.Sequential(
            nn.Linear(5023 * 3, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, z_dim), nn.LeakyReLU(0.2),
        )

        map_hidden_dim = 128
        self.network = nn.ModuleList(
            [nn.Linear(z_dim, map_hidden_dim)] +
            [nn.Linear(map_hidden_dim, map_hidden_dim) for i in range(3)]
        )

        self.output = nn.Linear(map_hidden_dim, 101)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.output.weight *= 0.25

        # self.feat_vol_reso = cfg.model.vol_feat_reso
        # self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))

        # grouping configuration
        # self.n_offset_groups = cfg.model.n_offset_groups
        # self.register_buffer("group_centers", self.build_dense_grid(self.grid_reso*2))
        # self.group_centers = self.group_centers.reshape(1,-1,3)

        # 2DGS model
        # self.sh_dim = (cfg.model.sh_degree+1)**2*3
        # self.scaling_dim, self.rotation_dim = 2, 4
        # self.opacity_dim = 1
        # self.out_dim = self.sh_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim

        # self.K = cfg.model.K
        # vol_embedding_out_dim = cfg.model.vol_embedding_out_dim
        # self.decoder = Decoder(vol_embedding_out_dim, self.sh_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, self.K,
        #                       cnn_dim=cfg.model.cnn_dim)
        # self.gs_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1)

        # parameters initialization
        # self.opacity_shift = -2.1792
        # self.voxel_size = 2.0/(self.grid_reso*2)
        # self.scaling_shift = np.log(0.5*self.voxel_size/3.0)

        # self.has_cnn = cfg.model.cnn_dim > 0
        # assert cfg.model.cnn_dim <= 13
        # if self.has_cnn:
        #    self.cnn = Upsampler()
        # self.cnn_dim = cfg.model.cnn_dim

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'), dim=-1)
        grid = (grid + 0.5) / reso * 2 - 1
        return grid.reshape(reso, reso, reso, 3) * self.scene_size

    def add_pos_enc_patches(self, src_inps, img_feats, n_views_sel, batch):

        h, w = src_inps.shape[-2:]
        # src_ixts = batch['tar_ixt'][:,:n_views_sel].reshape(-1,3,3)
        # src_w2cs = batch['tar_w2c'][:,:n_views_sel].reshape(-1,4,4)

        # img_wh = torch.tensor([w,h], device=self.device)
        # point_img,_ = projection(self.volume_grid, src_w2cs, src_ixts)
        # point_img = (point_img+ 0.5)/img_wh*2 - 1.0

        # viewing direction
        rays = batch['tar_rays_down'][:, :n_views_sel]
        feats_dir = self.ray_to_plucker(rays).reshape(-1, *rays.shape[2:])
        feats_dir = torch.cat((rsh_cart_3(feats_dir[..., :3]), rsh_cart_3(feats_dir[..., 3:6])), dim=-1)

        # query features
        img_feats = torch.einsum('bchw->bhwc', img_feats)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum('bhwc->bchw', img_feats)

        # n_channel = img_feats.shape[1]
        # feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        ## img features
        # feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)

        return img_feats

    def _check_mask(self, mask):
        ratio = torch.sum(mask) / np.prod(mask.shape)
        if ratio < 1e-3:
            mask = mask + torch.rand(mask.shape, device=self.device) > 0.8
        elif ratio > 0.5 and self.training:
            # avoid OMM
            mask = mask * torch.rand(mask.shape, device=self.device) > 0.5
        return mask

    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):

        points = points[mask]
        n_points = points.shape[0]

        h, w = img_ref.shape[-2:]
        src_ixts = batch['tar_ixt'][idx, :n_views_sel].reshape(-1, 3, 3)
        src_w2cs = batch['tar_w2c'][idx, :n_views_sel].reshape(-1, 4, 4)

        img_wh = torch.tensor([w, h], device=self.device)
        point_xy, point_z = projection(points, src_w2cs, src_ixts)
        point_xy = (point_xy + 0.5) / img_wh * 2 - 1.0

        imgs_coarse = torch.cat((renderings['image'], renderings['acc_map'].unsqueeze(-1), renderings['depth']), dim=-1)
        imgs_coarse = torch.cat((img_ref, torch.einsum('bhwc->bchw', imgs_coarse)), dim=1)
        feats_coarse = F.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False).view(n_views_sel, -1,
                                                                                                   n_points).to(
            imgs_coarse)

        z_diff = (feats_coarse[:, -1:] - point_z.view(n_views_sel, -1, n_points)).abs()

        point_feats = torch.cat((feats_coarse[:, :-1], z_diff), dim=1)  # [...,_mask]

        return point_feats, mask

    def ray_to_plucker(self, rays):
        origin, direction = rays[..., :3], rays[..., 3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)

        # Calculate the moment vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)

        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment), dim=-1)

    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5 * self.scene_size / self.n_offset_groups
        centers = self.group_centers.unsqueeze(-2).expand(B, -1, K, -1).reshape(offset.shape) + offset * half_cell_size
        return centers

    def forward(self, batch, return_feature_map: bool = False):

        # B, N, H, W, C = batch['tar_rgb'].shape
        # if self.training:
        #    n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        # else:
        n_views_sel = 1  # self.cfg.n_views

        if return_feature_map:
            feat_map = None

        ## build 3D volume
        # TODO add plucker coordinates back
        # img_feats = self.add_pos_enc_patches(_inps, img_feats, n_views_sel, batch) # B n_views_sel C H W

        # decoding
        # img_feats = self.vol_decoder(img_feats)  # b c h w
        # img_feats = img_feats.permute(0, 2, 3, 1)
        # img_feats = self.activation(self.embed_mlp(img_feats)).permute(0, 3, 1, 2) # b c h w

        verts = batch['template_verts']
        verts = verts.reshape(verts.shape[0], -1)

        x = self.vertex_encoder(verts)
        enc = x
        # verts = verts.reshape(verts.shape[0], -1)
        for i_layer, layer in enumerate(self.network):
            # if i_layer == 0:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            # else:
            #    x = F.leaky_relu(layer(torch.cat([x, enc], dim=-1)), negative_slope=0.2)

        flame_code_pred = self.output(x)

        if return_feature_map:
            return flame_code_pred, feat_map
        return flame_code_pred

# if __name__ == '__main__':
