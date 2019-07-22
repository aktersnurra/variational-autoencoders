import torch
import torch.nn.functional as F


def loss_function(x, x_reconstructed, mu, logvar, batch_size, img_size=784):

    # measure of how well the VAE at reconstructing input from the latent space
    BCE = F.binary_cross_entropy(x_reconstructed, x.view(-1, img_size), reduction='sum')

    # Kullback-Leibler Divergence
    # Tells us how much a learnt distribution deviate from another
    # In the case of VAE from the paper Auto-Encoding Variational Bayes by Kingma et al.
    # we measure the learnt distribution from the unit Gaussian

    # D_{KL} = 0.5 \cdot \sum(1 + logvar - \mu^2 - \sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD







