import time

from train import init_cfg
from provider.dataset import TrainingDataset

cfg = init_cfg()
dataset = TrainingDataset(
    cfg.train_dataset,  # 配置文件
    cfg.dataset,    # [REAL275 | CAMERA25]
    cfg.mod,    # [r|ts]
    resolution=cfg.resolution,
    ds_rate=cfg.ds_rate,
    num_img_per_epoch=cfg.num_mini_batch_per_epoch * cfg.train_dataloader.bs    # 5000 * 48
)

print(dataset[0]['category_label'])
