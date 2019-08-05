import torch
import torch.nn as nn
from utils import initialize_weights
from utils.gelu import GELU


class InferenceNetwork(nn.Module):
    def __init__(self, params):
        super(InferenceNetwork, self).__init__()
        self.params = params
        self.fc = nn.Linear(in_features=self.params.input_dim, out_features=self.params.hidden_dim[0])
        self.fc_mu = nn.Linear(in_features=self.params.hidden_dim[0], out_features=self.params.hidden_dim[1])
        self.fc_logvar = nn.Linear(in_features=self.params.hidden_dim[0], out_features=self.params.hidden_dim[1])
        self.activation_fn = GELU()
        initialize_weights(self)

    def forward(self, x):
        x = x.view(-1, self.params.input_dim)
        h1 = self.activation_fn(self.fc(x))
        mu = self.activation_fn(self.fc_mu(h1))
        logvar = self.activation_fn(self.fc_logvar(h1))
        return mu, logvar


class GenerativeNetwork(nn.Module):
    def __init__(self, params):
        super(GenerativeNetwork, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(in_features=self.params.hidden_dim[1], out_features=self.params.hidden_dim[0])
        self.fc2 = nn.Linear(in_features=self.params.hidden_dim[0], out_features=self.params.input_dim)
        self.activation_fn = GELU()
        initialize_weights(self)

    def forward(self, z):
        h3 = self.activation_fn(self.fc1(z))
        out = torch.sigmoid(self.fc2(h3))
        return out


class VariationalAutoencoder(nn.Module):
    def __init__(self, params):
        super(VariationalAutoencoder, self).__init__()
        self.params = params
        self.inference_network = InferenceNetwork(params=params)
        self.generative_network = GenerativeNetwork(params=params)

    @staticmethod
    def _reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.inference_network(x)
        z = self._reparameterization(mu, logvar)
        x_reconstructed = self.generative_network(z)
        return x_reconstructed, mu, logvar


