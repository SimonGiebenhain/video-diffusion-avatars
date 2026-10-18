# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
import torch


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None):
    config = {} if config is None else config
    # skip the random initialisation: every parameter is overwritten by the checkpoint
    with skip_model_initialization():
        model = model_class(**config)
    # DiskMap reads only the tensors this model asks for (one file can hold several models, e.g. the
    # Wan2.2-Animate shards hold the DiT and the animate adapter)
    state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
    # some files need their keys renamed / filtered for the model class
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)
    else:
        state_dict = {i: state_dict[i] for i in state_dict}
    model.load_state_dict(state_dict, assign=True)
    model = model.to(dtype=torch_dtype, device=device)
    return model.eval()
