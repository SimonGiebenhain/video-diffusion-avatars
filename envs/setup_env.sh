#!/bin/bash
# ONE conda environment `vda` for generation AND preprocessing. Run from the repository root, on a
# machine with a GPU + driver for CUDA 12.8 (the CUDA extensions are compiled here).
#
#   generation   : Python 3.12, torch 2.10.0+cu128 and envs/requirements_generate.txt -- byte-for-byte
#                  the versions of the env that produced the reference renders. Every one of them is
#                  passed to pip as a constraint while the preprocessing stack is added, so none changes.
#   preprocessing: pixel3dmm (FLAME fits), nvdiffrast + pytorch3d built against torch 2.10,
#                  face_alignment, diffusers (X-NeMo encoder): envs/requirements_preprocess.txt.
#
# Host compiler: any gcc 9..14 works with CUDA 12.8; override with CC/CXX/CUDAHOSTCXX if the default
# gcc is newer (e.g. CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12 bash envs/setup_env.sh).
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_NAME=${VDA_ENV_NAME:-vda}       # VDA_ENV_NAME=<name> picks another env name
CU=nvidia/label/cuda-12.8.0
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"

# ---------------------------------------------------------------- python + CUDA 12.8 toolkit (nvcc, headers)
conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.12.13 pip
conda install -y -n "$ENV_NAME" --override-channels -c $CU -c conda-forge python=3.12.13 \
    $CU::cuda-nvcc $CU::cuda-cccl $CU::cuda-cudart-dev $CU::cuda-nvrtc-dev $CU::cuda-profiler-api \
    $CU::libcublas-dev $CU::libcusparse-dev $CU::libcusolver-dev $CU::libcurand-dev
conda activate "$ENV_NAME"
# the label packages keep their headers under targets/; torch's JIT builds (nvdiffrast) need them on CPATH
conda env config vars set -n "$ENV_NAME" CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/targets/x86_64-linux/include"
conda deactivate; conda activate "$ENV_NAME"

# ---------------------------------------------------------------- generation (unchanged)
# DO NOT install flash-attn / sage-attention / xformers (see requirements_generate.txt)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r "$REPO/envs/requirements_generate.txt"
CONSTRAINTS="$CONDA_PREFIX/vda_generation_constraints.txt"
pip freeze | grep -v -e '^-e' -e ' @ ' > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"          # every later pip call keeps the generation versions

# ---------------------------------------------------------------- preprocessing
pip install -r "$REPO/envs/requirements_preprocess.txt"
pip install --no-build-isolation chumpy==0.70
pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.5;8.0;8.6+PTX"} FORCE_CUDA=1 \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
pip install --no-deps -e "$REPO/third_party/pixel3dmm"
# this repository as the editable package `video-diffusion-avatars` (import vda from other repositories). --no-deps: the
# environment above already pins every dependency.
pip install --no-deps -e "$REPO"
# pixel3dmm preprocessing models: facer, PIPNet + its nms extension, FLAME 2020 (asks for your FLAME
# credentials) with the FLAME masks from the MICA repository, and the pixel3dmm uv / normal networks;
# see third_party/pixel3dmm/README.md
(cd "$REPO/third_party/pixel3dmm" && bash install_preprocessing_pipeline.sh)

# ---------------------------------------------------------------- checks
pip check
FREEZE="$CONDA_PREFIX/vda_freeze_check.txt"           # one freeze, then grep the file (a pipe into grep -q breaks under pipefail)
pip freeze > "$FREEZE"
grep -v '^#' "$CONSTRAINTS" | while read -r l; do grep -qixF "$l" "$FREEZE" || { echo "generation pin changed: $l"; exit 1; }; done
python -c "import torch, transformers, peft; from pytorch3d.ops import knn_points; import nvdiffrast.torch, face_alignment, diffusers, pixel3dmm, vda.generate, vda.preprocess.conditioning; \
print('vda env ok: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
echo "done. Next: download the weights (README, section Weights)."
