"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import logging
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn

from .custom_optim import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print = logger.info


def build_optimizer(
    model,
    optimizer_name: str,
    lr: float,
    weight_decay=0.0,
    momentum=0.0,
    sgd_dampening=0.0,
    sgd_nesterov=False,
    rmsprop_alpha=0.99,
    adam_beta1=0.9,
    adam_beta2=0.999,
    staged_lr: Optional[Dict] = None,
    lookahead=False,
):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """

    if optimizer_name not in AVAI_OPTIMS:
        raise ValueError(f"Unsupported optim: {optimizer_name}. Must be one of {AVAI_OPTIMS}")

    if staged_lr is not None:
        if not isinstance(model, nn.Module):
            warnings.warn(
                "When staged_lr is True, model given to "
                "build_optimizer() must be an instance of nn.Module."
                "You should reconstruct the param_groups manully"
            )
            param_groups = model
        else:
            if isinstance(model, nn.DataParallel):
                model = model.module

            param_groups = build_staged_lr_param_groups(model, lr, **staged_lr)

    else:
        if isinstance(model, nn.Module):
            param_groups = {"params": model.parameters(), "lr": lr, "init_lr": lr, "name": model.__class__.__name__}
        else:
            param_groups = model

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optimizer_name == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optimizer_name == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optimizer_name == "ranger":
        optimizer = Ranger(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optimizer_name == "ranger21":
        raise NotImplementedError
        optimizer = Ranger21(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            num_epochs=max_epochs,
            num_batches_per_epoch=num_batches_per_epoch,
        )
    elif optimizer_name == 'adai':
        logger.info("betas of adai/adaiw are fixed to (0.1, 0.99).")
        optimizer = Adai(
            params=param_groups,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adaiw':
        logger.info("betas of adai/adaiw are fixed to (0.1, 0.99).")
        optimizer = AdaiW(
            params=param_groups,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"optim: {optimizer_name} not in {AVAI_OPTIMS}")

    if lookahead and optimizer_name not in ("ranger", "ranger21"):
        optimizer = Lookahead(optimizer)
        print("use bare lookahead.")

    return optimizer


def build_staged_lr_param_groups(model, lr, new_layers, base_lr_mult, new_lr_mult):
    if isinstance(new_layers, list):
        if len(new_layers) == 0:
            warnings.warn("new_layers is empty, therefore, staged_lr is useless")

    if isinstance(new_layers, str):
        if new_layers is None:
            warnings.warn("new_layers is empty, therefore, staged_lr is useless")
        new_layers = [new_layers]

    base_params = []
    base_params_names = []
    new_params = []
    new_params_names = []

    for name, params in model.named_parameters():
        if any([name.startswith(prefix) for prefix in new_layers]):
            new_params.append(params)
            new_params_names.append(name)
        else:
            base_params.append(params)
            base_params_names.append(name)
    print(f"Num base_params_names: {len(base_params_names)} ({base_params_names[:5]}...)")
    print(f"Num new_params_names: {len(new_params_names)} ({new_params_names[:5]}...)")

    param_groups = [
        {
            "params": base_params,
            "lr": lr * base_lr_mult,
            "init_lr": lr * base_lr_mult,
            "name": f"{model.__class__.__name__}_base",
        },
        {
            "params": new_params,
            "lr": lr * new_lr_mult,
            "init_lr": lr * new_lr_mult,
            "name": f"{model.__class__.__name__}_new",
        },
    ]
    return param_groups
