import os
import sys
import time
import random
import logging
import argparse

import torch
import gorilla

from utils.solver import test_func
from utils.logger import init_logger
from utils.evaluation_utils import evaluate
from provider.dataset import TestingDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))


def init_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--test_epoch", type=int, default=0, help="test epoch")
    args = parser.parse_args()
    
    cfg = gorilla.Config.fromfile(args.config)
    cfg.gpus = args.gpus
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    cfg.test_epoch = args.test_epoch
    cfg.log_dir = os.path.join('log', cfg.dataset, cfg.exp_name, 'results')
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    return cfg


def init_model(cfg):
    from model.ts.net import Net as ts_Net
    ts_model = ts_Net(cfg.n_cls)
    from model.r.net import Net as r_Net
    r_model = r_Net(cfg.resolution, cfg.ds_rate)
    if len(cfg.gpus) > 1:
        ts_model = torch.nn.DataParallel(
            ts_model, range(len(cfg.gpus.split(","))))
        r_model = torch.nn.DataParallel(
            r_model, range(len(cfg.gpus.split(","))))
    ts_model = ts_model.cuda()
    r_model = r_model.cuda()
    # TODO: adjust load checkpoint
    checkpoint = os.path.join('/data4/lj/PoseEstimation/log/common/ts.pth')
    gorilla.solver.load_checkpoint(model=ts_model, filename=checkpoint)
    checkpoint = os.path.join('log', cfg.dataset, cfg.exp_name, 'r', 'epoch_' + str(cfg.test_epoch) + '.pth')
    gorilla.solver.load_checkpoint(model=r_model, filename=checkpoint)
    return ts_model, r_model


def init_data(cfg):
    dataset = TestingDataset(
        cfg.test_dataset,
        cfg.dataset,
        cfg.resolution
    )
    dataloder = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.test_dataloader.bs,
        num_workers=cfg.test_dataloader.num_workers,
        shuffle=cfg.test_dataloader.shuffle,
        drop_last=cfg.test_dataloader.drop_last
    )
    return dataloder


def run():
    cfg = init_cfg()
    logger = init_logger(
        level_print=logging.INFO,
        level_save=logging.INFO,
        path_file=cfg.log_dir+"/testing_logger.log"
    )
    
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    logger.info(f'start test logging... epoch: {cfg.test_epoch}')
    logger.info(f'using gpu: {cfg.gpus}')
    start = time.time()
    ts_model, r_model = init_model(cfg)
    logger.info(f"load model successfully. cost time: {time.time() - start}s")
    start = time.time()
    dataloder = init_data(cfg)
    logger.info(f"load data successfully. cost time: {time.time() - start}s")

    save_path = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        test_func(ts_model, r_model, dataloder, save_path)
    evaluate(save_path, logger)
    logger.info("end test logging...")


if __name__ == "__main__":
    run()
