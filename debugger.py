from modules import glimpse_network, BasicBlock
from modules import retina
from data_loader import get_train_valid_loader
from config import get_config

import torch

def test_glimpse():
    config, unparsed = get_config()
    train_loader, _ = get_train_valid_loader(
        config.data_dir, config.batch_size,
        config.random_seed, config.valid_size,
        config.shuffle, config.show_sample)

    img, label = next(iter(train_loader))


    r = retina(g=8, k=1, s=1)
    g = glimpse_network(block=BasicBlock, h_g=128, h_l=32, h_s=128, g=8, 
    k=1, s=1, c=3)
    
    l_t_prev = torch.rand(config.batch_size, 2).uniform_(-1,1)
    size_t_prev = torch.rand(config.batch_size, 5).uniform_(0,1)
    
    #print(g.feature_extractors["size_32"])
    g_t = g(img, l_t_prev, size_t_prev)

    return g_t

def test_retina():
    l_t_prev = torch.rand(config.batch_size, 2).uniform_(-1, 1)

    size_t_prev = torch.rand(config.batch_size, 5).uniform_(0,1)
    
    options = torch.Tensor([32, 16, 8, 4, 2, 1]).long()
    size_t_prev = options[torch.argmax(size_t_prev, 1)]

    x = r.extract_patch(img, l_t_prev, size_t_prev)
    
    return x

def test_tensor_idx():
    y =torch.Tensor([32, 16, 8, 4, 2, 1])
    #y = [y for _ in range(10)]
    #y = torch.stack(y)
    print(y)

    x = torch.rand(10, 5)
    idx = torch.argmax(x, 1).view(-1, 1)
    print(idx)
    z = y[idx]
    w = (z == 16).squeeze().nonzero()
    print(z, w.numpy().flatten().tolist())
   
x = torch.rand((2, 3))
print(x)
print(x.repeat(2, 2)) 
