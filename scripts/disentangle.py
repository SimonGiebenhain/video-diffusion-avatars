"""Split X-NeMo codes into a view-invariant expression part and a view part (code: vda/disentangler.py).

    python scripts/disentangle.py <CODES.npy> --out <Z.npz>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vda.disentangler import main  # noqa: E402

if __name__ == "__main__":
    main()
