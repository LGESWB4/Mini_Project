import argparse
import json
import os
import re
import glob
from importlib import import_module
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from losses.base_loss import get_loss_fn
from trainer.trainer import Trainer
from utils.util import ensure_dir, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base_config.json", help="Config file path")
    
    args, _ = parser.parse_known_args()    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.device == "cuda", "CUDA is not available! Please run on a CUDA-enabled device."
    print(args)
    return args


def increment_path(path, exist_ok=False):
    """자동으로 저장 폴더 경로 증가 (예: runs/exp → runs/exp1, exp2)"""
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(r"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def load_model(model_name, ckpt_path, device):
    model_module_name = "model." + model_name.lower() + "_custom"
    model_module = getattr(import_module(model_module_name), model_name)
    model = model_module().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def main(args):
    if args.is_wandb:
        wandb.init(project="rsp_classification", name=args.name, config=vars(args))

    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(args.model_dir, args.name))
    ensure_dir(save_dir)

    IMAGE_ROOT = os.path.join(args.data_dir, "train")

    device = args.device
    if args.ckpt != "None":
        exp_path = os.path.join("./outputs", args.ckpt)
        ckpt_path = os.path.join(exp_path, "best_epoch.pth")
        model = load_model(args.model, ckpt_path, device)
    else:
        model_name = "model.custom_model"
        model_module = getattr(import_module(model_name), args.model)
        model = model_module().to(device)

    dataset_module = getattr(import_module("datasets.base_dataset"), args.dataset)
    train_dataset = dataset_module(root_dir=IMAGE_ROOT)
    valid_dataset = dataset_module(root_dir=IMAGE_ROOT.replace("train", "val"))
    
    # -- augmentation
    IMG_SIZE = (300, 400)
    transform_module = getattr(import_module("datasets.augmentation"), args.augmentation)  # default: TrainAugmentation
    tr_transform = transform_module(img_size=IMG_SIZE, is_train=True)
    val_transform = transform_module(img_size=IMG_SIZE, is_train=False)

    train_dataset.set_transform(tr_transform)
    valid_dataset.set_transform(val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size // 2, shuffle=False, num_workers=2, drop_last=False)

    criterion = [get_loss_fn(cr) for cr in args.criterion]  # CrossEntropyLoss, FocalLoss, LabelSmoothingCrossEntropy

    opt_module = getattr(import_module("torch.optim"), args.optimizer["type"])
    optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), **dict(args.optimizer["args"]))

    sche_module = getattr(import_module("torch.optim.lr_scheduler"), args.lr_scheduler["type"])
    scheduler = sche_module(optimizer, **dict(args.lr_scheduler["args"]))

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        args_dict = vars(args)
        args_dict["model_dir"] = save_dir
        json.dump(args_dict, f, ensure_ascii=False, indent=4)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        save_dir,
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=valid_loader,
        lr_scheduler=scheduler,
    )

    trainer.train()


# python train.py --config ./configs/queue/MobileNetV3_CE.json
if __name__ == "__main__":
    args = parse_args()
    main(args)
