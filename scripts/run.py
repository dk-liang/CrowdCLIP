import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--config", "-c", action="append", type=str, default=[])
parser.add_argument("--seed", "-s", type=int, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--test_only", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument(
    "--cfg_options",
    default=[],
    nargs=argparse.REMAINDER,
    help="modify config options using the command-line",
)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import os.path as osp
import pdb
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from crowdclip.runner.data import RegressionDataModule
from crowdclip.runner.runner import Runner
from crowdclip.utils.logging import get_logger, setup_file_handle_for_all_logger

logger = get_logger(__name__)


def main(cfg: DictConfig):
    pl.seed_everything(cfg.runner_cfg.seed, True)
    output_dir = Path(cfg.runner_cfg.output_dir)
    setup_file_handle_for_all_logger(str(output_dir / "run.log"))

    callbacks = load_callbacks(output_dir)
    

    deterministic = True
    logger.info(f"`deterministic` flag: {deterministic}")

    trainer = pl.Trainer(
        callbacks=callbacks,
        deterministic=deterministic,
        **OmegaConf.to_container(cfg.trainer_cfg),
    )
    if cfg.trainer_cfg.fast_dev_run is True:
        from IPython.core.debugger import set_trace

        set_trace()

    runner = None
    regression_datamodule = RegressionDataModule(**OmegaConf.to_container(cfg.data_cfg))    #load dataset
    # Training
    #please wait. The code is coming soon


    # Testing
    ckpt_paths = list((output_dir / "ckpts").glob("*.ckpt"))
    if len(ckpt_paths) == 0:
        logger.info("zero shot")
        if runner is None:
            # print('cfg.runner_cfg:', cfg.runner_cfg)
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=regression_datamodule)
        logger.info(f"End zero shot.")

    for ckpt_path in ckpt_paths:
        logger.info(f"Start testing ckpt: {ckpt_path}.")

        # no need to load weights in runner wrapper
        for k in cfg.runner_cfg.load_weights_cfg.keys():
            cfg.runner_cfg.load_weights_cfg[k] = None
        cfg.runner_cfg.ckpt_path = str(ckpt_path)

        if runner is None:
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))   #load model

        runner = runner.load_from_checkpoint(str(ckpt_path), **OmegaConf.to_container(cfg.runner_cfg))  #load checkpoint

        trainer.test(model=runner, datamodule=regression_datamodule)    #test

        logger.info(f"End testing ckpt: {ckpt_path}.")


def load_callbacks(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "ckpts").mkdir(exist_ok=True, parents=True)

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val_mae_metric",
            dirpath=str(output_dir / "ckpts"),
            filename="{epoch:02d}-{val_mae_metric:.4f}",
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode="min",
            save_weights_only=True,
        )
    )
    return callbacks


def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)

    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"

    return output_dir


def get_version(path: Path):
    versions = os.listdir(path)
    v = [-1]
    for i in versions:
        if 'version' in i:
            v.append(int(i.split('_')[1]))
    v.sort()
    return v[-1] + 1
    


def parse_cfg(args, instantialize_output_dir=True):
    print('args.config:', args.config)
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    # print('cfg:', cfg)
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create())

    # Setup output_dir
    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if instantialize_output_dir:
        if not args.test_only:
            output_dir = setup_output_dir_for_training(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    seed = args.seed if args.seed is not None else cfg.runner_cfg.seed
    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=args.debug),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


if __name__ == "__main__":
    cfg = parse_cfg(args, instantialize_output_dir=True)

    logger.info("Start.")
    main(cfg)
    logger.info("End.")
