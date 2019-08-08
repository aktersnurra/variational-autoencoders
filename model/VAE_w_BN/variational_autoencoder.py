import torch
import torch.nn as nn
from utils import initialize_weights
from utils.gelu import GELU


class InferenceNetwork(nn.Module):
    def __init__(self, params):
        super(InferenceNetwork, self).__init__()
        self.params = params
        self.fc = nn.Linear(in_features=self.params.input_dim, out_features=self.params.hidden_dim)
        self.fc_mu = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)
        self.activation_fn = GELU()
        self.batch_norm_h = nn.BatchNorm1d(self.params.hidden_dim)
        self.batch_norm_mu = nn.BatchNorm1d(self.params.latent_dim)
        self.batch_norm_logvar = nn.BatchNorm1d(self.params.latent_dim)

        initialize_weights(self)

    def forward(self, x):
        x = x.view(-1, self.params.input_dim)
        h1 = self.batch_norm_h(self.activation_fn(self.fc(x)))
        mu = self.batch_norm_mu(self.activation_fn(self.fc_mu(h1)))
        logvar = self.batch_norm_logvar(self.activation_fn(self.fc_logvar(h1)))
        return mu, logvar


class GenerativeNetwork(nn.Module):
    def __init__(self, params):
        super(GenerativeNetwork, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(in_features=self.params.latent_dim, out_features=self.params.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.input_dim)
        self.activation_fn = GELU()
        self.batch_norm = nn.BatchNorm1d(self.params.hidden_dim)
        initialize_weights(self)

    def forward(self, z):
        h3 = self.batch_norm(self.activation_fn(self.fc1(z)))
        out = torch.sigmoid(self.fc2(h3))
        return out


class VariationalAutoencoder(nn.Module):
    def __init__(self, params):
        super(VariationalAutoencoder, self).__init__()
        self.params = params
        self.inference_network = InferenceNetwork(params=params)
        self.generative_network = GenerativeNetwork(params=params)

    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(torch.Size([1, self.params.hidden_dim]))
        return self.decode(eps).view(self.params.num_examples_to_generate, 1, 28, 28)

    @staticmethod
    def reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        mu, logvar = self.inference_network(x)
        return mu, logvar

    def decode(self, z):
        return self.generative_network(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z


