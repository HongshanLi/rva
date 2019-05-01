from modules import glimpse_network, BasicBlock
from modules import retina
from data_loader import get_train_valid_loader
from config import get_config

import torch

config, unparsed = get_config()
train_loader, _ = get_train_valid_loader(
    config.data_dir, config.batch_size,
    config.random_seed, config.valid_size,
    config.shuffle, config.show_sample)

img, label = next(iter(train_loader))


retina = retina(g=8, k=1, s=1)
g = glimpse_network(block=BasicBlock, h_g=128, h_l=128, g=8, k=1, s=1, c=3)


l_t_prev = torch.rand(config.batch_size, 2).uniform_(-1, 1)

x = g(img, l_t_prev)
print(x.shape)

