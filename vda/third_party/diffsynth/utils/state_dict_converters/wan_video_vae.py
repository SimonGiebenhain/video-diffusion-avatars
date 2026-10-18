# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
def WanVideoVAEStateDictConverter(state_dict):
    state_dict_ = {}
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    for name in state_dict:
        state_dict_['model.' + name] = state_dict[name]
    return state_dict_