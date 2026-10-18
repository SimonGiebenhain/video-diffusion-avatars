"""Render the conditioning (proxy render, skeleton, expression codes) for a reference, a driving sequence and a
camera path (code: vda/preprocess/conditioning.py).

    python render_conditioning.py --ref <NAME> --driving <DRV> --camera over_the_top_zoom --out_dir cache/cond/<RUN>
"""
from vda.preprocess.conditioning import main

if __name__ == "__main__":
    main()
