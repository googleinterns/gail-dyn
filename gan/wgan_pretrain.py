import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from .wgan_models import Generator, Discriminator

import pickle

# to run, python -m gan.wgan_pretrain.py

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)

gen_input_dim = 11+3
gen_latent_dim = gen_input_dim
gen_output_dim = 11

dis_input_dim = 11+3+11

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Configure data loader
def combine_gen_x_y_from_pickle(pathname):
    with open(pathname, "rb") as handle:
        saved_file = pickle.load(handle)
    XY = []
    for traj_idx, traj_tuples in saved_file.items():
        XY.extend(traj_tuples)
    XY = np.array(XY)
    return XY[:, :14], XY[:, 14:25]     # x: (st, at), y:(st1)


X, Y = combine_gen_x_y_from_pickle("hopper_3_ave_dx_normalfloor_n200.pkl")
# TODO: append zeros to X columns for pre-training
X = np.concatenate((X, np.zeros((X.shape[0], gen_latent_dim))), axis=1)
dataset = TensorDataset(Tensor(X), Tensor(Y))
gen_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# # Optimizers
# optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
# optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

# Loss for pretraining
loss_fn = torch.nn.L1Loss(reduction='sum')
# Pre-training
learning_rate = 1e-4
optimizer_G_pre = torch.optim.Adam(generator.parameters(), lr=learning_rate)
for epoch in range(80):
    running_loss = 0.0
    for i, (x_batch, y_batch) in enumerate(gen_loader):
        optimizer_G_pre.zero_grad()
        # forward + backward + optimize
        y_pred = generator(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer_G_pre.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("finished pre-training")
torch.save(generator.state_dict(), "G_pre.pt")

# # ----------
# #  Training
# # ----------
#
# batches_done = 0
# for epoch in range(opt.n_epochs):
#
#     for i, (imgs, _) in enumerate(dataloader):
#
#         # Configure input
#         real_imgs = Variable(imgs.type(Tensor))
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Sample noise as generator input
#         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#
#         # Generate a batch of images
#         fake_imgs = generator(z).detach()
#         # Adversarial loss
#         loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
#
#         loss_D.backward()
#         optimizer_D.step()
#
#         # Clip weights of discriminator
#         for p in discriminator.parameters():
#             p.data.clamp_(-opt.clip_value, opt.clip_value)
#
#         # Train the generator every n_critic iterations
#         if i % opt.n_critic == 0:
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#
#             optimizer_G.zero_grad()
#
#             # Generate a batch of images
#             gen_imgs = generator(z)
#             # Adversarial loss
#             loss_G = -torch.mean(discriminator(gen_imgs))
#
#             loss_G.backward()
#             optimizer_G.step()
#
#             print(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
#             )
#
#         if batches_done % opt.sample_interval == 0:
#             save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
#         batches_done += 1
