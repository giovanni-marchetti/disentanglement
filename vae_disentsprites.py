from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np


batch_size = 128
epochs = 200
log_interval = 10

latent_dim = 4
pixels = 4096

beta = 1
gamma = 1000   #disentalgment factor
translation = False  

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


size_data = 200000
sprites = np.load('imgs.npy', mmap_mode = 'c')
sprites = torch.FloatTensor(sprites[ : size_data])
#sprites = F.interpolate(sprites.view(size_data, 1, 64, 64), size = (28, 28)) #downsampling
factors = np.load('latents_values.npy', mmap_mode = 'c')
factors = factors[ : size_data, 2 : ]
factors[ : , 0] = (factors [ : , 0] - 0.75) / 0.25       #normalizing to [-1,1]
factors[ : , 1] = (factors [ : , 1] - np.pi) / np.pi
factors[ : , 2] = (factors [ : , 2] - 0.5) / 0.5
factors[ : , 3] = (factors [ : , 3] - 0.5) / 0.5
factors = torch.FloatTensor(factors)

#print(factors[90000:90020])
#print(np.shape(factors))



train_loader = torch.utils.data.DataLoader(sprites,
                                            batch_size=batch_size) #no shuffling
train_factors = torch.utils.data.DataLoader(factors,
                                            batch_size=batch_size)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(pixels, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc32 = nn.Linear(1200, latent_dim)
        self.fc31 = nn.Linear(1200, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 1200)
        self.fc5 = nn.Linear(1200, 1200)
        self.fc6 = nn.Linear(1200, pixels) 
        self.drop = nn.Dropout(p=0.2) 
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = self.drop(h)
        h = F.relu(self.fc5(h))
        h = self.drop(h)
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):    #Monte Carlo (with one sample) 
        mu, logvar = self.encode(x.view(-1, pixels))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z 


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, pixels), reduction='sum')
    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * KL_loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(zip(train_loader, train_factors)):
        
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        if translation: 
            disent = ((z - labels) * (z - labels)).sum(-1).var()
        else:
            disent = ((z - labels) * (z - labels)).sum(-1).mean()
            
        loss = loss + gamma * disent
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def latent_traversal(dim, num, base):   
    steps = np.arange(-2 * num, 0) / num
    sample = []
    for i in range(dim):
        tmp = np.full((2 * num, dim), base)
        for j in range(2 * num):
            tmp[j][i] += steps[j]
        sample.append(torch.FloatTensor(tmp))
    return sample
    
if __name__ == "__main__":
    sample = latent_traversal(latent_dim, 8, 1)
    for epoch in range(1, epochs + 1):
        train(epoch)
        if (epoch < 10) or (epoch % 10 == 0): 
            for i in range(4):
                with torch.no_grad():
                    img = model.decode(sample[i])
                    save_image(img.view(16, 1, 64, 64),
                               'results_id/sample_' + str(epoch) + '_' + str(i) + '.png', nrow = 16)
            sample1 = torch.randn((64, latent_dim))
            with torch.no_grad():
                img = model.decode(sample1)
                save_image(img.view(64, 1, 64, 64),
                           'results_id/random_' + str(epoch) + '.png')



               
       
