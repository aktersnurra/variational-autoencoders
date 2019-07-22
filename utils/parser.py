import argparse


def argument_parser():
    """
    A method for parsing arguments passed via the command line for training a Variational Autoencoder. The default
    arguments will load a fully connected Variational Autoencoder that will be trained on the MNIST dataset.

    Returns
    -------
    Returns a argparse object.

    """
    parser = argparse.ArgumentParser(description='Train Variational Autoencoder')

    parser.add_argument('--data_dir',
                        nargs="?",
                        default='data/mnist',
                        help="Directory containing the dataset")

    parser.add_argument('--model_dir',
                        nargs="?",
                        default='experiments/fully_connected',
                        help="Directory containing params.json")

    parser.add_argument('--restore_file',
                        default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'

    parser.add_argument('--dataloader',
                        default='mnist',
                        help='Specify the dataloader to use')

    return parser.parse_args()

