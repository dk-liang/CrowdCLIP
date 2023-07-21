import errno
import os
import os.path as osp
import pickle
import shutil
import warnings
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from clip.clip import tokenize
from torchvision import transforms

from crowdclip.utils.logging import get_logger, print_log

logger = get_logger(__name__)
print = lambda x: print_log(x, logger=logger)

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
    "open_all_layers",
    "open_specified_layers",
    "count_num_param",
    "load_pretrained_weights",
    "init_network_weights",
]


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name="",
    topk=3,
    filter_prefix="",
):
    r"""Save checkpoint.
    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)

    if topk > 0:
        torch.save(state, fpath)
        print(f'Checkpoint saved to "{fpath}"')

        # save current model name
        checkpoint_file = osp.join(save_dir, "checkpoint")
        checkpoint = open(checkpoint_file, "a")
        checkpoint.write(f"{osp.basename(fpath)}\n")
        checkpoint.close()

        # only save topk
        with open(checkpoint_file, "r") as f:
            lines = f.readlines()
        if len(lines) > topk:
            with open(checkpoint_file + ".tmp", "w") as f:
                for i in range(len(lines) - topk, len(lines)):
                    f.write(lines[i])
            # from IPython.core.debugger import set_trace
            # set_trace()

            for file_path in lines[: len(lines) - topk]:
                file_path = osp.join(save_dir, file_path.strip("\n"))
                if osp.isfile(file_path):
                    os.remove(file_path)

            shutil.copy(checkpoint_file + ".tmp", checkpoint_file)
            os.remove(checkpoint_file + ".tmp")

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        # shutil.copy(fpath, best_fpath)
        best_state = OrderedDict()
        if filter_prefix:
            for k, v in state["state_dict"].items():
                if filter_prefix not in k:
                    best_state[k] = v
        else:
            best_state = state["state_dict"]
        torch.save({"epoch": state["epoch"], "state_dict": best_state}, best_fpath)
        print(f'Best checkpoint saved to "{best_fpath}"')


def load_checkpoint(fpath):
    r"""Load checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError(f'File is not found at "{fpath}"')

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)

    except Exception:
        print(f'Unable to load checkpoint from "{fpath}"')
        raise

    return checkpoint


def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    r"""Resume training from a checkpoint.
    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.
    Args:
        fdir (str): directory where the model was saved.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (Scheduler, optional): an Scheduler.
    Returns:
        int: start_epoch.
    Examples::
        >>> fdir = 'log/my_model'
        >>> scheckpoint_filetart_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
    """
    start_epoch = 0
    checkpoint_file = osp.join(fdir, "checkpoint")
    if not osp.exists(checkpoint_file):
        with open(checkpoint_file, "w") as f:
            pass
        return start_epoch

    with open(checkpoint_file, "r") as checkpoint:
        model_names = checkpoint.readlines()
        if len(model_names) == 0:
            return start_epoch
        model_name = model_names[-1].strip("\n")
        fpath = osp.join(fdir, model_name)

    print(f'Loading checkpoint from "{fpath}"')
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded model weights")

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded optimizer")

    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler")

    start_epoch = checkpoint["epoch"]
    print(f"Previous epoch: {start_epoch}")

    return start_epoch


def set_bn_to_eval(m):
    r"""Set BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def open_all_layers(model):
    r"""Open all layers in model for training.
    Examples::
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Open specified layers in model for training while keeping
    other layers frozen.
    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.
    Examples::
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(model, layer), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Count number of parameters in a model.
    Args:
        model (nn.Module): network model.
    Examples::
        >>> model_size = count_num_param(model)
    """
    return sum(p.numel() for p in model.parameters())


def load_pretrained_weights(model: nn.Module, weight_path, fix_pretrain_weights=False):
    r"""Load pretrianed weights to model.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
    Examples::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    if model is None:
        print("model is not instantialized.")
        return

    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if checkpoint.get("epoch"):
        print(f"load from epoch: {checkpoint.get('epoch')}!")

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    print(f"len of matched ckpt state_dict: {len(new_state_dict)}")
    print(f"len of model state_dict: {len(model_dict)}")
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if fix_pretrain_weights:
        cnt = 0
        for name, params in model.named_parameters():
            if name in matched_layers:
                cnt += 1
                freeze_param(params)
        print(f"Fix pre-trained-weights of {weight_path}")
        print(f"Num of fixed pre-trained-weights layers: {cnt}")

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(weight_path)
        )
        raise NameError(f"No matched layers for checkpoint: {weight_path}.")
    else:
        print(f'Successfully loaded pretrained weights from "{weight_path}"')
        if len(discarded_layers) > 0:

            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".format(discarded_layers)
            )


def init_network_weights(model, init_type="normal", gain=0.02):
    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


def get_transforms(
    input_transforms,
    input_resize,
    input_size,
    pixel_mean,
    pixel_std,
):
    transform_train = []
    transform_test = []
    if "random_resized_crop" in input_transforms:
        transform_train.append(transforms.Resize(input_resize))
        transform_test.append(transforms.Resize(input_resize))

        transform_train.append(transforms.RandomCrop(input_size))
        transform_test.append(transforms.CenterCrop(input_size))
    if "random_hflip" in input_transforms:
        transform_train.append(transforms.RandomHorizontalFlip())

    transform_train.append(transforms.ToTensor())
    transform_test.append(transforms.ToTensor())

    if "normalize" in input_transforms:
        transform_train.append(transforms.Normalize(pixel_mean, pixel_std))
        transform_test.append(transforms.Normalize(pixel_mean, pixel_std))

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def set_requires_grad(module, requires_grad):
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = requires_grad
    elif isinstance(module, nn.parameter.Parameter):
        module.requires_grad = requires_grad
    else:
        raise TypeError(f"The type of the module is wrong: {type(module)}")

    return None


def freeze_param(module):
    if module is None:
        print("model is not instantialized.")
        return
    set_requires_grad(module, False)


def unfreeze_param(module):
    if module is None:
        print("model is not instantialized.")
        return
    set_requires_grad(module, True)


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = (1,) + start.size()
    w_size = (steps,) + (1,) * start.dim()
    out_size = (steps,) + start.size()

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
