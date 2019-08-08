import torch
import torch.nn.functional as F


def loss_function(x, x_reconstructed, mu, logvar, z, use_mse=False):
    # measure of how well the VAE at reconstructing input from the latent space
    # E[p(X|z)]
    batch_size = x.size(0)
    if use_mse:
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / batch_size
    else:
        reconstruction_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum') / batch_size

    # Kullback-Leibler Divergence
    # Tells us how much a learnt distribution deviate from another
    # In the case of VAE from the paper Auto-Encoding Variational Bayes by Kingma et al.
    # we measure the learnt distribution from the unit Gaussian

    # D_{KL} = 0.5 \cdot \sum(1 + logvar - \mu^2 - \sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld_loss = torch.mean(kld_loss)

    loss = reconstruction_loss + kld_loss

    return {'loss': loss, 'Kullback-Leibler_Divergence': kld_loss, 'Reconstruction_Loss': reconstruction_loss}

