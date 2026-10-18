"""Per-frame body pose of a driving video with SAM 3D Body, for prepare_driving.py --motion driver
(code: vda/preprocess/body_motion.py). Run it in the SAM 3D Body environment, from the checkout:

    python scripts/prepare_body_motion.py --name <DRV> --sam3d <SAM3D_REPO> [--no_fov]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vda.preprocess.body_motion import main  # noqa: E402

if __name__ == "__main__":
    main()
