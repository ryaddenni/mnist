import math
# new gan.Py
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn


# Setup for distributed training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# Cleanup after training
def cleanup():
    dist.destroy_process_group()


# Transformation for MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', 
                          train=True,
                          download=True,
                          transform=transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


# Dataset information
print(train_ds.data.shape)
print(train_ds.targets.shape)
print(train_ds.classes)
print(train_ds.data[0])
print(train_ds.targets[0])
print(train_ds.data[0].max())
print(train_ds.data[0].min())
print(train_ds.data[0].float().mean())
print(train_ds.data[0].float().std())


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(32, out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)                         
        x = self.fc3(x)
        x = self.leaky_relu3(x)        
        x = self.dropout(x)
        logit_out = self.fc4(x)
        return logit_out


# Define Generator
class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(64, 128)
        self.relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)        
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)        
        x = self.dropout(x)
        x = self.fc4(x)
        tanh_out = self.tanh(x)
        return tanh_out


# Loss functions
def real_loss(predicted_outputs, loss_fn, device):
    # Ensure the predicted_outputs have the correct shape
    predicted_outputs = predicted_outputs.view(-1)  # Flatten the output tensor
    targets = torch.ones_like(predicted_outputs, device=device)  # Create target tensor of ones (real labels)
    real_loss = loss_fn(predicted_outputs, targets)
    return real_loss

def fake_loss(predicted_outputs, loss_fn, device):
    # Ensure the predicted_outputs have the correct shape
    predicted_outputs = predicted_outputs.view(-1)  # Flatten the output tensor
    targets = torch.zeros_like(predicted_outputs, device=device)  # Create target tensor of zeros (fake labels)
    fake_loss = loss_fn(predicted_outputs, targets)
    return fake_loss


# Training function
def train_mnist_gan(rank, world_size, d, g, d_optim, g_optim, loss_fn, dl, n_epochs, device):
    setup(rank, world_size)

    z_size = 100
    fixed_z = torch.randn(16, z_size).to(device)
    d.to(device)
    g.to(device)

    device = torch.device(f"cuda:{rank}")

    # Wrap the models with DDP
    d = DDP(d, device_ids=[rank])
    g = DDP(g, device_ids=[rank])

    # Distributed sampler for data loading
    train_sampler = DistributedSampler(dl, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=dl, shuffle=False, batch_size=64, num_workers=4, pin_memory=True, sampler=train_sampler)

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch}/{n_epochs}] on rank {rank}...")
        train_sampler.set_epoch(epoch)

        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            real_images = (real_images * 2) - 1  # Normalize to [-1, 1]

            # Discriminator step
            d_optim.zero_grad()
            d_real_loss = real_loss(d(real_images), loss_fn, device)
            z = torch.randn(real_images.size(0), z_size).to(device)
            fake_images = g(z)
            d_fake_loss = fake_loss(d(fake_images), loss_fn, device)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Generator step
            g_optim.zero_grad()
            z = torch.randn(real_images.size(0), z_size).to(device)
            fake_images = g(z)
            g_loss = real_loss(d(fake_images), loss_fn, device)
            g_loss.backward()
            g_optim.step()

            if rank == 0:
                print(f"Epoch {epoch}, Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}")

        dist.barrier()  # Ensure synchronization between processes after each epoch

    cleanup()



# Main function to spawn processes
def main():
    world_size = 2  # Number of GPUs
    n_epochs = 100

    # Initialize models and optimizers
    d = Discriminator(784, 1)
    g = Generator(100, 784)
    d_optim = optim.Adam(d.parameters(), lr=0.002)
    g_optim = optim.Adam(g.parameters(), lr=0.002)
    loss_fn = nn.BCEWithLogitsLoss()

    # Use spawn to launch multiple processes for DDP
    spawn(train_mnist_gan, args=(world_size, d, g, d_optim, g_optim, loss_fn, train_ds, n_epochs, device), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
