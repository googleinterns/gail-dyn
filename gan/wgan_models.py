import torch.nn as nn
import torch.nn.functional as F
import torch

gen_input_dim = 11+3
gen_latent_dim = gen_input_dim
gen_output_dim = 11

dis_input_dim = 11+3+11


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(gen_input_dim+gen_latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, gen_output_dim),
        )

    def forward(self, gen_in):
        next_state = self.model(gen_in)
        return next_state


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dis_input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, dis_in):
        validity = self.model(dis_in)
        return validity