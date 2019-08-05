import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from utils import initialize_weights
from utils.gelu import GELU


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, channel, height, width):
        super(UnFlatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class InferenceNetwork(nn.Module):
    def __init__(self, params):
        super(InferenceNetwork).__init__()
        self.params = params

        inference_layers = [
            nn.Conv2d(in_channels=self.params.channels[0],
                      out_channels=self.params.channels[1],
                      kernel_size=self.params.kernel_size[0],
                      stride=self.params.stride[0],),
            nn.BatchNorm2d(self.params.channels[1]),
            GELU(),
            nn.Conv2d(in_channels=self.params.channels[1],
                      out_channels=self.params.channels[2],
                      kernel_size=self.params.kernel_size[1],
                      stride=self.params.stride[1]),
            nn.BatchNorm2d(self.params.channels[2]),
            GELU(),
            nn.Conv2d(in_channels=self.params.channels[2],
                      out_channels=self.params.channels[3],
                      kernel_size=self.params.kernel_size[2],
                      stride=self.params.stride[2]),
            nn.BatchNorm2d(self.params.channels[3]),
            GELU(),
            Flatten(),
        ]

        self.encoder = nn.Sequential(*inference_layers)

        self.activation_fn = GELU()
        self.fc_mu = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)

        initialize_weights(self)

    def forward(self, x):
        h1 = self.encoder(x)
        mu = self.activation_fn(self.fc_mu(h1))
        logvar = self.activation_fn(self.fc_logvar(h1))

        return mu, logvar


class GenerativeNetwork(nn.Module):
    def __init__(self, params, heigth=4, width=4):
        super(GenerativeNetwork).__init__()
        self.params = params
        self.sigmoid = nn.Sigmoid()

        generative_layers = [
            nn.Linear(in_features=self.params.latent_dim, out_features=self.params.hidden_dim),
            GELU(),
            UnFlatten(self.params.channels[3], heigth, width),
            nn.ConvTranspose2d(in_channels=self.params.channels[2],
                               out_channels=self.params.channels[3],
                               kernel_size=self.params.kernel_size[2],
                               stride=self.params.stride[2]),
            nn.BatchNorm2d(self.params.channels[3]),
            GELU(),
            nn.ConvTranspose2d(in_channels=self.params.channels[3],
                               out_channels=self.params.channels[4],
                               kernel_size=self.params.kernel_size[3],
                               stride=self.params.stride[3]),
            nn.BatchNorm2d(self.params.channels[4]),
            GELU(),
            nn.ConvTranspose2d(in_channels=self.params.channels[4],
                               out_channels=self.params.channels[5],
                               kernel_size=self.params.kernel_size[4],
                               stride=self.params.stride[4]),
        ]

        self.decoder = nn.Sequential(*generative_layers)

        initialize_weights(self)

    def forward(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = self.sigmoid(logits)
            return probs

        return logits


class VariationalAutoencoder(nn.Module):
    def __init__(self, params):
        super(VariationalAutoencoder, self).__init__()
        self.params = params
        self.inference_network = InferenceNetwork(params)
        self.generative_network = GenerativeNetwork(params)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def sample(self, eps=None):
        if eps is None:
            eps = self.normal.sample(torch.Size([1, self.params.hidden_dim]))
        return self.decode(eps, apply_sigmoid=True)

    @staticmethod
    def reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        mu, logvar = self.inference_network(x)
        return mu, logvar

    def decode(self, z, apply_sigmoid=False):
        return self.generative_network(z, apply_sigmoid)



