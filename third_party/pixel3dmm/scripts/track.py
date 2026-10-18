import os
import wandb

from omegaconf import OmegaConf
from pixel3dmm.tracking.tracker import Tracker
from pixel3dmm import env_paths

def main(cfg):
    tracker = Tracker(cfg)
    tracker.run()

if __name__ == '__main__':
    base_conf = OmegaConf.load(f'{env_paths.CODE_BASE}/configs/tracking.yaml')

    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)

    main(cfg)