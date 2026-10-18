# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
def WanVideoDiTStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        name_ = name
        if name_.startswith("model."):
            name_ = name_[len("model."):]
        state_dict_[name_] = state_dict[name]
    return state_dict_
