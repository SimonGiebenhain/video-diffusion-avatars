"""Generate the video from a conditioning folder with the full or the fast model (code: vda/generate.py).

    python generate.py --cond cache/cond/<RUN> --out <OUT.mp4> [--model fast]
"""
from vda.generate import main

if __name__ == "__main__":
    main()
