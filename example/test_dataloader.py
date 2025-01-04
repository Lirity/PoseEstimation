import time

from train import init, load_data

cfg, _ = init()
dataloaders = load_data(cfg)

start = time.time()

st = time.time()
for i, data in enumerate(dataloaders["train"]):
    print(f'{i}: cost: {time.time() - st}s')
    st = time.time()

print(f'all cost: {time.time() - start}s')
