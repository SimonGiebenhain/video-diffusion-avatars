"""Download what VDA needs (code: vda/download.py).

    python download.py weights     # our weights (~6.5 GB, Hugging Face) into weights/, md5-checked
    python download.py base        # the public WAN2.2-Animate-14B files (~48 GB) into weights/base/
    python download.py examples    # the 25 examples and the example driving videos into examples/ (~220 MB)
"""
from vda.download import main

if __name__ == "__main__":
    main()
