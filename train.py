import numpy as np
import os
import argparse

# Dataset related
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

# DDP Libaries
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import model
from model import LeNet

def reduce_loss(tensor, local_rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if local_rank == 0:
            tensor /= world_size

def trainer():
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        # DistributedSampler deterministically shuffle data
        # The order of shuffled data will be the same for all epochs if we do not call set_epoch
        sampler.set_epoch(epoch)
        for idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs)
            loss = ce_loss(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # reduce loss
            reduce_loss(loss, local_rank, world_size)
            # By default, we only log the results on gpu 0
            if idx % 50 == 0 and local_rank == 0:
                print('Epoch: {} step: {} loss: {}'.format(epoch, idx, loss.item()))
                
    # Testing loop
    model.eval()
    with torch.no_grad():
        cnt = 0
        total = len(test_loader.dataset)
        for imgs, labels in test_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs)
            predict = torch.argmax(output, dim=1)
            cnt += (predict == labels).sum().item()
    
    if local_rank == 0:
        print('eval accuracy: {}'.format(cnt / total))
    return model

if __name__ == '__main__':
    # Parse configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Local rank refers to the id of the gpu [0,1,2,...]
    local_rank = int(os.environ["LOCAL_RANK"])
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    
    # Initialize dataset
    trainset = MNIST(root='dataset',
                     download=True,
                     train=True,
                     transform=transforms.ToTensor())

    testset = MNIST(root='dataset',
                   download=True,
                   train=False,
                   transform=transforms.ToTensor())
    
    batch_size = 128
    # Initialize dataloaders
    sampler = DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=sampler,
                              num_workers=4,
                              pin_memory=True)

    test_loader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    
    # Initialize model
    model = LeNet()
    model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Perform training
    trainer()