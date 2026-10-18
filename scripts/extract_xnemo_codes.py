"""X-NeMo expression codes [T, 512] of a video (code: vda/preprocess/xnemo_codes.py).

    python scripts/extract_xnemo_codes.py --video <VIDEO.mp4> --out <CODES.npy>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vda.preprocess.xnemo_codes import main  # noqa: E402

if __name__ == "__main__":
    main()
