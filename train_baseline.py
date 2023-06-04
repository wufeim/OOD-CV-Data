import argparse
from datetime import datetime
import logging
import os
import random

import numpy as np
import torch

from src.config import load_config
from src.evaluate import pose_error
from src.utils import setup_logging, set_seed
from src.models.utils import construct_class_by_name


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline 3D pose estimation model for OOD-CV')
    parser.add_argument('--config', type=str, default='resnet.yaml')
    parser.add_argument('--save_dir', type=str, default=f'resnet50_oodcv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def evaluate(cfg, dataloader, model):
    pose_errors = []
    for i, sample in enumerate(dataloader):
        pred = model.evaluate(sample)
        _err = pose_error(sample, pred['final'][0])
        pose_errors.append(_err)
    pose_errors = np.array(pose_errors)
    
    acc6 = np.mean(pose_errors<np.pi/6) * 100
    acc18 = np.mean(pose_errors<np.pi/18) * 100
    mederr = np.median(pose_errors) / np.pi * 180
    return {'acc6': acc6, 'acc18': acc18, 'mederr': mederr}


def train(cfg):
    train_dataset = construct_class_by_name(**cfg.dataset, data_type='train', category='all')
    if cfg.dataset.sampler is not None:
        train_dataset_sampler = construct_class_by_name(
            **cfg.dataset.sampler, dataset=train_dataset, rank=0, num_replicas=1,
            seed=cfg.training.random_seed)
        shuffle = False
    else:
        train_dataset_sampler = None
        shuffle = True
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.workers,
        sampler=train_dataset_sampler)
    val_dataset = construct_class_by_name(**cfg.dataset, data_type='val', category='all')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)
    logging.info(f"Number of training images: {len(train_dataset)}")
    logging.info(f"Number of validation images: {len(val_dataset)}")

    model = construct_class_by_name(
        **cfg.model, cfg=cfg.model, cate='all', mode='train')

    logging.info("Start training")
    for epo in range(cfg.training.total_epochs):
        num_iterations = int(cfg.training.scale_iterations_per_epoch * len(train_dataloader))
        for i, sample in enumerate(train_dataloader):
            if i >= num_iterations:
                break
            loss_dict = model.train(sample)

        if (epo + 1) % cfg.training.log_interval == 0:
            logging.info(
                f"[Epoch {epo+1}/{cfg.training.total_epochs}] {model.get_training_state()}"
            )

        if (epo + 1) % cfg.training.ckpt_interval == 0:
            torch.save(model.get_ckpt(epoch=epo+1, cfg=cfg.asdict()), os.path.join(cfg.args.save_dir, "ckpts", f"model_{epo+1}.pth"))
            results = evaluate(cfg, val_dataloader, model)
            logging.info(f'[Validation {epo+1}] acc@pi/6={results["acc6"]:.2f} acc@pi/18={results["acc18"]:.2f} mederr={results["mederr"]:.2f}')

        model.step_scheduler()


def main():
    args = parse_args()

    setup_logging(args.save_dir)
    set_seed(args.seed)

    cfg = load_config(args, load_default_config=False, log_info=False)
    logging.info(args)
    logging.info(cfg)

    train(cfg)


if __name__ == '__main__':
    main()
