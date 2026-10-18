# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
import glob, os
from dataclasses import dataclass
from typing import Union
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download


@dataclass
class ModelConfig:
    """A model file: an explicit `path`, or `model_id` + `origin_file_pattern` resolved under
    $DIFFSYNTH_MODEL_BASE_PATH (default ./models) and downloaded unless $DIFFSYNTH_SKIP_DOWNLOAD=true,
    from $DIFFSYNTH_DOWNLOAD_SOURCE (modelscope | huggingface, default modelscope)."""
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    local_model_path: str = None
    
    def check_input(self):
        if self.path is None and self.model_id is None:
            raise ValueError("""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")
    
    def parse_original_file_pattern(self):
        if self.origin_file_pattern in [None, "", "./"]:
            return "*"
        elif self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        else:
            return self.origin_file_pattern
        
    def parse_download_source(self):
        return os.environ.get('DIFFSYNTH_DOWNLOAD_SOURCE') or "modelscope"
        
    def parse_skip_download(self):
        return os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD', '').lower() == "true"

    def download(self):
        origin_file_pattern = self.parse_original_file_pattern()
        downloaded_files = glob.glob(origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))
        download_source = self.parse_download_source()
        if download_source.lower() == "modelscope":
            snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_file_pattern=origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False
            )
        elif download_source.lower() == "huggingface":
            hf_snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_patterns=origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False
            )
        else:
            raise ValueError("`download_source` should be `modelscope` or `huggingface`.")
        
    def require_downloading(self):
        if self.path is not None:
            return False
        return not self.parse_skip_download()
    
    def reset_local_model_path(self):
        if os.environ.get('DIFFSYNTH_MODEL_BASE_PATH') is not None:
            self.local_model_path = os.environ.get('DIFFSYNTH_MODEL_BASE_PATH')
        elif self.local_model_path is None:
            self.local_model_path = "./models"

    def download_if_necessary(self):
        self.check_input()
        self.reset_local_model_path()
        if self.require_downloading():
            self.download()
        if self.path is None:
            if self.origin_file_pattern in [None, "", "./"]:
                self.path = os.path.join(self.local_model_path, self.model_id)
            else:
                self.path = glob.glob(os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern))
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]
