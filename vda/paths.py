"""Where VDA finds its files. Every path in the package comes from here; nothing depends on the current directory
except paths you pass on the command line and the ./cache fallback below.

Source checkout (`pip install -e .`, the recommended install): repo_root() is the checkout, and by default
  <checkout>/weights/           $VDA_WEIGHTS   model weights (README "Weights")
  <checkout>/cache/             $VDA_CACHE     preprocessed references, driving sequences, conditionings
  <checkout>/examples/                         downloaded examples (python download.py examples)
  <checkout>/third_party/pixel3dmm/  $PIXEL3DMM_CODE_BASE   pixel3dmm (preprocessing only)
Installed without the checkout (a wheel): set $VDA_WEIGHTS; the cache falls back to ./cache in the current
directory; preprocessing (reference / driving / conditioning) needs the checkout or $PIXEL3DMM_CODE_BASE.

weights/
  vda_full.safetensors          full model adapter (fine-tuned WAN-Animate adapter)
  vda_fast_base.safetensors     base adapter of the distilled model (+ its teacher)
  vda_fast_student.safetensors  distilled causal student (LoRA on DiT + face fuser)
  xnemo_disentangler.pt         X-NeMo expression disentangler (frozen in-graph pre-encoder)
  text_emb_cache.pt             cached umt5 prompt embeddings (skips the 11 GB text encoder)
  driving_bank/                 shipped driving sequence (+ canonical body for --motion driver)
  third_party/                  X-NeMo motion encoder weights (own driving videos)
  base/                         DIFFSYNTH_MODEL_BASE_PATH: public WAN files (python download.py base)
"""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ASSETS = PACKAGE / "assets"                      # package data: FLAME/MHR index files, camera presets

FULL_ADAPTER = "vda_full.safetensors"
FAST_BASE_ADAPTER = "vda_fast_base.safetensors"
FAST_STUDENT = "vda_fast_student.safetensors"
DISENTANGLER = "xnemo_disentangler.pt"
TEXT_CACHE = "text_emb_cache.pt"


def repo_root():
    """The source checkout this package runs from (editable install / plain checkout), or None."""
    root = PACKAGE.parent
    if (root / "pyproject.toml").is_file() and (root / "third_party").is_dir():
        return root
    return None


def weights_dir() -> Path:
    if os.environ.get("VDA_WEIGHTS"):
        return Path(os.environ["VDA_WEIGHTS"]).expanduser().resolve()
    root = repo_root()
    if root is not None:
        return root / "weights"
    raise SystemExit("VDA weights not found: set VDA_WEIGHTS=<WEIGHTS_DIR> (README, section Weights)")


def weight(name: str) -> str:
    p = weights_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"missing weight file {p} (set $VDA_WEIGHTS or see README 'Weights')")
    return str(p)


def cache_dir() -> Path:
    """$VDA_CACHE, else <checkout>/cache, else ./cache in the current directory (installed without checkout)."""
    if os.environ.get("VDA_CACHE"):
        return Path(os.environ["VDA_CACHE"]).expanduser().resolve()
    root = repo_root()
    return (root / "cache") if root is not None else Path("cache").resolve()


def driving_state(arg) -> str:
    """A driving sequence given on the command line: a state .npz, the name of a shipped sequence
    (<weights>/driving_bank/<name>.npz, e.g. nersemble8) or the name of a video prepared with prepare_driving.py
    (<cache>/driving/<name>/state.npz)."""
    p = Path(arg).expanduser()
    if p.is_file():
        return str(p)
    for c in (weights_dir() / "driving_bank" / f"{arg}.npz", cache_dir() / "driving" / str(arg) / "state.npz"):
        if c.is_file():
            return str(c)
    raise SystemExit(f"driving sequence {arg!r} not found: pass a .npz, a name in {weights_dir() / 'driving_bank'} "
                     f"or a name prepared with prepare_driving.py ({cache_dir() / 'driving'})")


def reference_dir(arg) -> str:
    """A reference given on the command line: a directory with flame_shape.npy, a shape .npy, or the name of a
    reference prepared with prepare_reference.py (<cache>/refs/<name>)."""
    p = Path(arg).expanduser()
    if p.exists():
        return str(p)
    c = cache_dir() / "refs" / str(arg)
    if (c / "flame_shape.npy").is_file():
        return str(c)
    raise SystemExit(f"reference {arg!r} not found: pass a directory / .npy, or a name prepared with prepare_reference.py "
                     f"({cache_dir() / 'refs'})")


def reference_image(arg) -> str:
    """A reference image given to python generate.py: an image file, a reference / example directory (its ref_916.png or
    ref.png) or the name of a reference prepared with prepare_reference.py."""
    p = Path(arg).expanduser()
    if p.is_file():
        return str(p)
    for d in (p, cache_dir() / "refs" / str(arg)):
        for f in ("ref_916.png", "ref.png"):
            if (d / f).is_file():
                return str(d / f)
    raise SystemExit(f"reference image {arg!r} not found: pass an image, a reference / example folder, or a name "
                     f"prepared with prepare_reference.py ({cache_dir() / 'refs'})")


def examples_dir() -> Path:
    """<checkout>/examples, else ./examples in the current directory."""
    root = repo_root()
    return (root / "examples") if root is not None else Path("examples").resolve()


def pixel3dmm_dir() -> Path:
    """The pixel3dmm code base used for preprocessing: $PIXEL3DMM_CODE_BASE, else <checkout>/third_party/pixel3dmm."""
    if os.environ.get("PIXEL3DMM_CODE_BASE"):
        p = Path(os.environ["PIXEL3DMM_CODE_BASE"]).expanduser().resolve()
    else:
        root = repo_root()
        p = root / "third_party" / "pixel3dmm" if root is not None else None
    if p is None or not (p / "scripts" / "track.py").is_file():
        raise SystemExit("preprocessing needs the source checkout (pip install -e .) or PIXEL3DMM_CODE_BASE "
                         f"pointing to third_party/pixel3dmm (got {p})")
    return p


def pixel3dmm_env(work=None) -> dict:
    """PIXEL3DMM_* variables for pixel3dmm (its env_paths reads them at import time): the code base, and the
    preprocessed-data / tracking-output folders under `work` (default <cache>/pixel3dmm)."""
    work = Path(work) if work is not None else cache_dir() / "pixel3dmm"
    return {"PIXEL3DMM_CODE_BASE": str(pixel3dmm_dir()),
            "PIXEL3DMM_PREPROCESSED_DATA": str(work / "preprocessed"),
            "PIXEL3DMM_TRACKING_OUTPUT": str(work / "tracking")}


def setup_pixel3dmm_import():
    """Before importing pixel3dmm in this process: point its env_paths at our copy. Explicit settings win over a
    personal ~/.config/pixel3dmm/.env, which env_paths would otherwise read."""
    os.environ.update(pixel3dmm_env())


def setup_base_model_env():
    """Point DiffSynth at weights/base unless the caller already configured it."""
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(weights_dir() / "base"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
