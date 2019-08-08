"""Generate images with the Variational Autoencoder"""
import os
import sys
import glob
import argparse
import imageio
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
from PIL import Image

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


def generate_images(model, model_dir, dataloader, params):
    log_dir = misc.create_log_dir(model_dir, generated_img=True)
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
                writer.add_image("generated_images_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), images, i)


def generate_and_save_images(model, epoch, test_input, img_dir):
    X_reconstructed = model.sample(test_input)
    X_reconstructed = X_reconstructed.cpu().detach()

    images = torchvision.utils.make_grid(X_reconstructed)
    filename = os.path.join(img_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    torchvision.utils.save_image(images, filename)


def generate_gif(img_dir, tbx_writer=None):
    anim_file = os.path.join(img_dir, 'cvae.gif')
    with imageio.get_writer(anim_file, mode='I', format='GIF', fps=5) as writer:
        filenames = glob.glob(os.path.join(img_dir, 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    if tbx_writer:
        gif_image = Image.open(anim_file)
        tbx_writer.add_image("animated_mnist_generation" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), gif_image)


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

    # Fetch the model
    if args.model_dir.split('/')[-1] == 'vae':
        from model.VAE.variational_autoencoder import VariationalAutoencoder
    elif args.model_dir.split('/')[-1] == 'vae_w_bn':
        from model.VAE_w_BN.variational_autoencoder import VariationalAutoencoder
    elif args.model_dir.split('/')[-1] == 'convolutional':
        from model.convolutional_VAE import VariationalAutoencoder

    model = VariationalAutoencoder(params).cuda() if params.cuda else VariationalAutoencoder(params)
    # Reload weights from the saved file
    misc.load_checkpoint(os.path.join(root, args.model_dir, args.restore_file + '.pth.tar'), model)
    # Generate images
    logging.info("\nGenerating images.\n")
    generate_images(model, model_dir, test_dl, params)