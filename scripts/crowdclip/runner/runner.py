import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from torch import nn

from crowdclip.models import MODELS
from crowdclip.models.crowdclip import CrowdCLIP
from crowdclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights
import numpy as np
import cv2
import time
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

logger = get_logger(__name__)
f_count = open("counting_result.txt", "w+")



class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss = 1.0,
            rank_loss = 1.0,
        ),
        ckpt_path="",
    ) -> None:
        super().__init__()
        # print('model_cfg:', model_cfg)
        self.module = MODELS.build(model_cfg)

        self.rank_loss = nn.MarginRankingLoss()
        self.loss_weights = loss_weights

        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.best_mae = 100000.0

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()

    # Running Steps
    def run_step(self, batch, batch_idx, phase='train'):

        x, y, file_name = batch
        new_x = x

        fname = os.path.basename(file_name[0])

        if phase == 'train':
            logits, count, *_ = self.module(new_x, 'train')
        elif phase == 'val' or phase == 'test':
            with torch.no_grad():
                logits, count, *_ = self.module(new_x, 'test')

        losses = self.compute_losses(logits, phase)

        loss = losses['rank_loss']


        metrics = self.compute_per_example_metrics(y, fname, count)

        return {"loss": loss, **losses, **metrics}

    def training_step(self, batch, batch_idx):
        # print('-----------------------train---------------------')
        outputs = self.run_step(batch, batch_idx, 'train')

        self.logging(outputs, "train", batch_idx, on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        # print('-----------------------val---------------------')
        outputs = self.run_step(batch, batch_idx, 'val')

        return outputs

    def test_step(self, batch, batch_idx):
        # print('-----------------------test---------------------')
        outputs = self.run_step(batch, batch_idx, 'test')

        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                if k == 'mse_metric':
                    stats[k] = torch.stack(_stats).mean().item()
                    stats[k] = np.sqrt(stats[k])
                else:
                    stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            if k == 'mae_metric':
                if stats[k] < self.best_mae:
                    self.best_mae = stats[k]
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        stats["best_mae"] = str(self.best_mae)
        del stats["loss"]
        del stats["rank_loss"]
        logger.info(json.dumps(stats))


    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, batch_idx, on_step=True, on_epoch=True):
        stats = defaultdict(list)
        for k, v in outputs.items():
            if self._valid_key(k):
                # print(f"{run_type}_{k}", v)
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)
                stats[k] = v.mean().item()
        if (batch_idx+1) % 300 == 0:
            del stats["mae_metric"]
            del stats["mse_metric"]
            logger.info(json.dumps(stats))

    # Loss & Metrics
    def compute_losses(self, logits, gather_type='train'):
        losses = {}

        z = torch.zeros(1).cuda()
        losses["rank_loss"] = z
        if gather_type == 'train':
            y1 = torch.ones(1).cuda()
            for i in range(logits.size()[0]):
                for j in range(i+1, logits.size()[0]):
                    losses["rank_loss"] += self.rank_loss(logits[j][j].unsqueeze(0), logits[i][j].unsqueeze(0), y1)
        return losses


    def compute_per_example_metrics(self, y, fname, count):
        predict_y = 0
        for i in count:
            predict_y += i

        mae = abs(predict_y - torch.sum(y))
        mse = abs(predict_y - torch.sum(y)) * abs(predict_y - torch.sum(y))
        y = y.type(torch.float32)
        # print('pred_count:{} gt_count:{} mae:{} mse:{}'.format(predict_y, torch.sum(y), mae, mse))
        fname = fname.split('_')[0] + '_' + fname.split('_')[1] + '.jpg'
        f_count.write('{} {} {} {} {}'.format(fname, predict_y, torch.sum(y), mae, mse))
        f_count.write('\n')

        mae = torch.abs(torch.tensor(predict_y).cuda() - torch.sum(y))
        mse = torch.abs(torch.tensor(predict_y).cuda() - torch.sum(y)) * torch.abs(torch.tensor(predict_y).cuda() - torch.sum(y))

        return {f"mae_metric": mae, "mse_metric": mse, "predict_y": predict_y}


    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        # print('param_dict_cfg:', param_dict_cfg)
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        return

    def build_param_dict(
        self,
        lr_image_encoder,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
    ):
        param_dict_ls = []

        if lr_image_encoder > 0 and self.module.image_encoder is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=self.module.image_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.image_encoder.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "image_encoder",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.image_encoder)")
            freeze_param(self.module.image_encoder)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.logit_scale)")
            freeze_param(self.module.logit_scale)
        return param_dict_ls


