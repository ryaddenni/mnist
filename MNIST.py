import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib as plt

# Setup for distributed training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

d_losses = []  # Discriminator losses
g_losses = []

# Cleanup after training
def cleanup():
    dist.destroy_process_group()

# Transformation for MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

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
        # we use LeakyRelu 
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
    predicted_outputs = predicted_outputs.view(-1)
    targets = torch.ones_like(predicted_outputs, device=device)
    return loss_fn(predicted_outputs, targets)

def fake_loss(predicted_outputs, loss_fn, device):
    predicted_outputs = predicted_outputs.view(-1)
    targets = torch.zeros_like(predicted_outputs, device=device)
    return loss_fn(predicted_outputs, targets)

# Training function
def train(rank, world_size, d, g, d_optim, g_optim, loss_fn, dl, n_epochs, device):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    z_size = 100
    d.to(device)
    g.to(device)

    # Wrap models with DDP
    d = DDP(d, device_ids=[rank])
    g = DDP(g, device_ids=[rank])

    # Distributed sampler for data loading
    train_sampler = DistributedSampler(dl, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        dataset=dl,
        batch_size=64,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    try:
        for epoch in range(n_epochs):
            if rank == 0:
                print(f"Epoch {epoch} started.", flush=True)

            train_sampler.set_epoch(epoch)

            for batch_idx, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)
                real_images = (real_images * 2) - 1  # Normalize to [-1, 1]

                # Discriminator step
                d_optim.zero_grad(set_to_none=True)
                z = torch.randn(real_images.size(0), z_size).to(device)
                fake_images = g(z)

                d_real_loss = real_loss(d(real_images), loss_fn, device)
                d_fake_loss = fake_loss(d(fake_images.detach()), loss_fn, device)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optim.step()

                # Generator step
                g_optim.zero_grad(set_to_none=True)
                z = torch.randn(real_images.size(0), z_size).to(device)
                fake_images = g(z)

                g_loss = real_loss(d(fake_images), loss_fn, device)
                g_loss.backward()
                g_optim.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                
                if rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}", flush=True)

            dist.barrier()
    finally:
        cleanup()

# Main function to spawn processes
def main():
    world_size = 2
    n_epochs = 500

    # Initialize dataset
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Initialize models and optimizers
    d = Discriminator(784, 1)
    g = Generator(100, 784)
    d_optim = optim.Adam(d.parameters(), lr=0.0002)
    g_optim = optim.Adam(g.parameters(), lr=0.0002)
    loss_fn = nn.BCEWithLogitsLoss()

    torch.multiprocessing.spawn(
        train,
        args=(world_size, d, g, d_optim, g_optim, loss_fn, train_ds, n_epochs, "cuda"),
        nprocs=world_size,
        join=True,
    )
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("Loss Curves")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()
   

if __name__ == "__main__":
    main()
