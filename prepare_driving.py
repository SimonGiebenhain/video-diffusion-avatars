"""Prepare a driving video: track the face (expression, jaw, gaze, blinks) and extract the expression codes;
cached in cache/driving/<DRV>/ (code: vda/preprocess/driving.py).

    python prepare_driving.py --video <DRIVER.mp4> --name <DRV>
"""
from vda.preprocess.driving import main

if __name__ == "__main__":
    main()
