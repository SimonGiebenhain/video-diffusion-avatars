"""Prepare a reference image: crop it to 720x1280 and fit its FLAME head shape; cached in cache/refs/<NAME>/
(code: vda/preprocess/reference.py).

    python prepare_reference.py --image <PHOTO.jpg> --name <NAME> [--min_conf 0.25]
"""
from vda.preprocess.reference import main

if __name__ == "__main__":
    main()
