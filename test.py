import os
import sys
import random
import logging
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

import torch
import gorilla

from utils.solver import test_func
from utils.logger import get_logger
from utils.evaluation_utils import evaluate
from model.ts.net import Net as ts_Net
from model.r.net import Net as r_Net
from provider.dataset import TestingDataset

def get_parser():
    parser = argparse.ArgumentParser(description="PoseEstimation")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="path to config file")
    parser.add_argument("--gpus", type=str, default="0", help="gpu id")
    parser.add_argument("--dataset", type=str, default="REAL275", help="[REAL275 | CAMERA25]")
    parser.add_argument("--test_epoch", type=int, default=0, help="test epoch")
    args_cfg = parser.parse_args()
    return args_cfg

def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.gpus = args.gpus
    cfg.dataset = args.dataset
    cfg.test_epoch = args.test_epoch
    cfg.log_dir = os.path.join('log', args.dataset, cfg.exp_name)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    cfg.save_path = os.path.join(cfg.log_dir, 'results')
    if not os.path.isdir(cfg.save_path):
        os.makedirs(cfg.save_path)
    logger = get_logger(level_print=logging.INFO, level_save=logging.INFO, path_file=cfg.save_path + "/testing_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)
    return cfg, logger

def load_model(cfg):
    ts_model = ts_Net(cfg.n_cls)
    r_model = r_Net(cfg.resolution, cfg.ds_rate)
    if len(cfg.gpus)>1:
        ts_model = torch.nn.DataParallel(ts_model, range(len(cfg.gpus.split(","))))
        r_model = torch.nn.DataParallel(r_model, range(len(cfg.gpus.split(","))))
    ts_model = ts_model.cuda() 
    r_model = r_model.cuda()
    checkpoint = os.path.join(cfg.log_dir, 'ts', 'epoch_' + 'test' + '.pth')
    gorilla.solver.load_checkpoint(model=ts_model, filename=checkpoint)
    checkpoint = os.path.join(cfg.log_dir, 'r', 'epoch_' + str(cfg.test_epoch) + '.pth')
    gorilla.solver.load_checkpoint(model=r_model, filename=checkpoint)
    return ts_model, r_model

def load_data(cfg):
    dataset = TestingDataset(cfg.test, cfg.dataset, cfg.resolution)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False
        )
    return dataloder

def run():
    cfg, logger = init()
    logger.info("start test logging...")
    ts_model, r_model = load_model(cfg)
    dataloder = load_data(cfg)
    save_path = os.path.join(cfg.save_path, 'epoch_' + str(cfg.test_epoch))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        test_func(ts_model, r_model, dataloder, save_path)
    evaluate(save_path, logger)
    logger.info("end test logging...")


if __name__ == "__main__":
    run()





    

    
