"""Block-seam colour fix for fast-model videos (code: vda/postprocess/deflicker.py); generate.py --deflicker runs it too.

    python scripts/deflicker.py <IN.mp4> <OUT.mp4>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vda.postprocess.deflicker import main  # noqa: E402

if __name__ == "__main__":
    main()
