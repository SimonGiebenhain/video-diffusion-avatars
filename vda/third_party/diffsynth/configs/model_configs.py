# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
# Trimmed for the video-diffusion-avatars release: only the entries for the files the release loads
# (Wan2.2-Animate-14B DiT + its animate adapter, Wan2.1 VAE, open-clip ViT-H image encoder, umt5-xxl text
# encoder for vda.text_cache). Model files are matched by state-dict hash (core/loader); entries are verbatim
# from DiffSynth-Studio.
MODEL_CONFIGS = [{'model_hash': '9c8818c2cbea55eca56c7b447df170da',
  'model_name': 'wan_video_text_encoder',
  'model_class': 'vda.third_party.diffsynth.models.wan_video_text_encoder.WanTextEncoder'},
 {'model_hash': 'ccc42284ea13e1ad04693284c7a09be6',
  'model_name': 'wan_video_vae',
  'model_class': 'vda.third_party.diffsynth.models.wan_video_vae.WanVideoVAE',
  'state_dict_converter': 'vda.third_party.diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter'},
 {'model_hash': '5941c53e207d62f20f9025686193c40b',
  'model_name': 'wan_video_image_encoder',
  'model_class': 'vda.third_party.diffsynth.models.wan_video_image_encoder.WanImageEncoder',
  'state_dict_converter': 'vda.third_party.diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter'},
 {'model_hash': '31fa352acb8a1b1d33cd8764273d80a2',
  'model_name': 'wan_video_dit',
  'model_class': 'vda.third_party.diffsynth.models.wan_video_dit.WanModel',
  'extra_kwargs': {'has_image_input': True,
                   'patch_size': [1, 2, 2],
                   'in_dim': 36,
                   'dim': 5120,
                   'ffn_dim': 13824,
                   'freq_dim': 256,
                   'text_dim': 4096,
                   'out_dim': 16,
                   'num_heads': 40,
                   'num_layers': 40,
                   'eps': 1e-06},
  'state_dict_converter': 'vda.third_party.diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter'},
 {'model_hash': '31fa352acb8a1b1d33cd8764273d80a2',
  'model_name': 'wan_video_animate_adapter',
  'model_class': 'vda.third_party.diffsynth.models.wan_video_animate_adapter.WanAnimateAdapter',
  'state_dict_converter': 'vda.third_party.diffsynth.utils.state_dict_converters.wan_video_animate_adapter.WanAnimateAdapterStateDictConverter'}]
