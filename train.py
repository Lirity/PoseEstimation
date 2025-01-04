import os
import sys
import time
import random
import logging
import argparse

import torch
import gorilla
import torch.utils
import torch.utils.data

from utils.solver import Solver
from utils.logger import init_logger
from provider.dataset import TrainingDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))


def init_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--mod", type=str, default="r")
    parser.add_argument("--checkpoint_epoch", type=int, default=-1)
    args = parser.parse_args()

    cfg = gorilla.Config.fromfile(args.config)
    cfg.gpus = args.gpus
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    cfg.mod = args.mod
    cfg.checkpoint_epoch = args.checkpoint_epoch
    cfg.log_dir = os.path.join('log', cfg.dataset, cfg.exp_name, cfg.mod)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    return cfg


def init_model(cfg):
    if cfg.mod == 'r':
        from model.r.net import Net
        from model.r.loss import Loss
        model = Net(cfg.resolution, cfg.ds_rate)
        loss = Loss(cfg.loss).cuda()
    elif cfg.mod == 'ts':
        from model.ts.net import Net
        from model.ts.loss import Loss
        model = Net(cfg.n_cls)
        loss = Loss(cfg.loss).cuda()
    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    # count_parameters = sum(gorilla.parameter_count(model).values())
    return model, loss


def init_data(cfg):
    dataset = TrainingDataset(
        cfg.train_dataset,
        cfg.dataset,
        cfg.mod,
        resolution=cfg.resolution,
        ds_rate=cfg.ds_rate,
        num_img_per_epoch=cfg.num_mini_batch_per_epoch * cfg.train_dataloader.bs,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers=int(cfg.train_dataloader.num_workers),
        shuffle=cfg.train_dataloader.shuffle,
        drop_last=cfg.train_dataloader.drop_last,
        pin_memory=cfg.train_dataloader.pin_memory,
    )
    dataloaders = {
        "train": dataloader,
    }
    return dataloaders


def run():
    cfg = init_cfg()
    logger = init_logger(
        level_print=logging.INFO,
        level_save=logging.INFO,
        path_file=cfg.log_dir+"/training_logger.log"
    )

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    logger.info("start train logging...")
    logger.info(f'using gpu: {cfg.gpus}')
    start = time.time()
    model, loss = init_model(cfg)
    logger.info(f"load model successfully. cost time: {time.time() - start}s")
    start = time.time()
    dataloaders = init_data(cfg)
    logger.info(f"load data successfully. cost time: {time.time() - start}s")

    Trainer = Solver(
        model=model,
        loss=loss,
        dataloaders=dataloaders,
        cfg=cfg,
        logger=logger,
    )
    Trainer.solve()
    logger.info("end train logging...")


if __name__ == "__main__":
    run()
