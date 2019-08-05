"""Generate images with the Variational Autoencoder"""
import os
import sys
import argparse
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

sys.path.append("..")
from utils import misc
from utils.parser import argument_parser


def show_images(images):
    """Transforms multiple images into a grid.

    Code referenced from http://hameddaily.blogspot.com/2018/12/yet-another-tutorial-on-variational.html

    Parameters
    ----------
    images

    Returns
    -------

    """
    images = torchvision.utils.make_grid(images)
    show_image(images[0])

def show_image(img):
    """Presents the generated images in a window.

    Code referenced from http://hameddaily.blogspot.com/2018/12/yet-another-tutorial-on-variational.html

    Parameters
    ----------
    img

    Returns
    -------

    """
    plt.imshow(img, cmap='gray')
    plt.show()


def generate_images(model, dataloader, params):
    log_dir = os.path.join(model_dir, 'runs')
    misc.create_dir(log_dir)
    with torch.no_grad() and SummaryWriter(log_dir) as writer:
        for i, (test_batch, _) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                test_batch = test_batch.cuda()
            # compute model output and loss
            X_reconstructed, mu, logvar = model(test_batch)

            if i % 31 == 0:
                X_reconstructed = X_reconstructed.view(params.batch_size, 1, 28, 28).cpu()
                images = torchvision.utils.make_grid(X_reconstructed)
                writer.add_image("Generated_images", images, i)


if __name__=='__main__':
    root = os.path.abspath('..')
    # Load the parameters from json file
    args = argument_parser()
    misc.tab_printer(args)
    model_dir = os.path.join(root, args.model_dir)
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = misc.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    misc.set_logger(os.path.join(root, args.model_dir, 'test.log'))

    # Create the input data pipeline
    logging.info("\nLoading the datasets...")

    # fetch dataloaders
    if args.dataloader == 'mnist':
        import data_loaders.mnist_data_loader as data_loader

    # types, data_dir, download, params
    data_dir = os.path.join(root, args.data_dir)
    dataloaders = data_loader.fetch_dataloader(types=['test'], data_dir=data_dir, download=False, params=params)
    test_dl = dataloaders['test']

    # Define the model and optimizer
    if 'fully_connected' in args.model_dir:
        from model.fully_connected_VAE.variational_autoencoder import VariationalAutoencoder

    model = VariationalAutoencoder(params).cuda() if params.cuda else VariationalAutoencoder(params)
    # Reload weights from the saved file
    misc.load_checkpoint(os.path.join(root, args.model_dir, args.restore_file + '.pth.tar'), model)
    # Generate images
    logging.info("\nGenerating images.\n")
    generate_images(model, test_dl, params)