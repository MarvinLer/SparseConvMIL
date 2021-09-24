__author__ = 'marvinler'

import argparse

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.utils.data

from dataset import Dataset
from model import instantiate_sparseconvmil


def _define_args():
    parser = argparse.ArgumentParser(description='SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance '
                                                 'Learning for Whole Slide Image Classification')

    parser.add_argument('--slide-parent-folder', type=str, default='sample_data', metavar='PATH',
                        help='path of parent folder containing preprocessed slides data')
    parser.add_argument('--slide-labels-filepath', type=str, default='sample_data/labels.csv', metavar='PATH',
                        help='path of CSV-file containing slide labels')

    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR', help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-6, metavar='R', help='weight decay')

    # Model parameters
    parser.add_argument('--tile-embedder', type=str, default='resnet18', metavar='MODEL', nargs='*',
                        help='type of resnet architecture for the tile embedder')
    parser.add_argument('--tile-embedder-pretrained', action='store_true', default=False,
                        help='use Imagenet-pretrained tile embedder architecture')
    parser.add_argument('--sparse-conv-n-channels-conv1', type=int, default=32,
                        help='number of channels of first convolution of the sparse-input CNN pooling')
    parser.add_argument('--sparse-conv-n-channels-conv2', type=int, default=32,
                        help='number of channels of first convolution of the sparse-input CNN pooling')
    parser.add_argument('--sparse-map-downsample', type=int, default=10, help='downsampling factor of the sparse map')
    parser.add_argument('--wsi-embedding-classifier-n-inner-neurons', type=int, default=32,
                        help='number of inner neurons for the WSI embedding classifier')

    # Dataset parameters
    parser.add_argument('--batch-size', type=int, default=2, metavar='SIZE',
                        help='number of slides sampled per iteration')
    parser.add_argument('--n-tiles-per-wsi', type=int, default=5, metavar='SIZE',
                        help='number of tiles to be sampled per WSI')

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')

    args = parser.parse_args()
    hyper_parameters = {
        'slide_parent_folder': args.slide_parent_folder,
        'slide_labels_filepath': args.slide_labels_filepath,
        'epochs': args.epochs,
        'lr': args.lr,
        'reg': args.reg,
        'tile_embedder': args.tile_embedder,
        'tile_embedder_pretrained': args.tile_embedder_pretrained,
        'sparse_conv_n_channels_conv1': args.sparse_conv_n_channels_conv1,
        'sparse_conv_n_channels_conv2': args.sparse_conv_n_channels_conv2,
        'sparse_map_downsample': args.sparse_map_downsample,
        'wsi_embedding_classifier_n_inner_neurons': args.wsi_embedding_classifier_n_inner_neurons,
        'batch_size': args.batch_size,
        'n_tiles_per_wsi': args.n_tiles_per_wsi,
    }

    return hyper_parameters


def get_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)


def perform_epoch(mil_model, dataloader, optimizer, loss_function):
    """
    Perform a complete training epoch by looping through all data of the dataloader.
    :param mil_model: MIL model to be trained
    :param dataloader: loader of the dataset
    :param optimizer: pytorch optimizer
    :param loss_function: loss function to compute gradients
    :return: (mean of losses, balanced accuracy)
    """
    proba_predictions = []
    ground_truths = []
    losses = []

    for data, locations, slides_labels, slides_ids in dataloader:
        data = data.cuda()
        locations = locations.cuda()
        slides_labels_cuda = slides_labels.cuda()

        optimizer.zero_grad()
        predictions = mil_model(data, locations)

        loss = loss_function(predictions, slides_labels_cuda)
        loss.backward()
        optimizer.step()

        # Store data for finale epoch average measures
        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(predictions.detach().cpu().numpy())
        ground_truths.extend(slides_labels.numpy())

    predicted_classes = np.argmax(proba_predictions, axis=1)
    return np.mean(losses), metrics.balanced_accuracy_score(ground_truths, predicted_classes)


def main(hyper_parameters):
    # Loads dataset and dataloader
    dataset = Dataset(hyper_parameters['slide_parent_folder'], hyper_parameters['slide_labels_filepath'],
                      hyper_parameters['n_tiles_per_wsi'])
    n_classes = dataset.n_classes
    dataloader = get_dataloader(dataset, hyper_parameters['batch_size'], True,
                                hyper_parameters['tile_embedder_pretrained'])

    # Loads MIL model, optimizer and loss function
    sparseconvmil_model = instantiate_sparseconvmil(hyper_parameters['tile_embedder'],
                                                    hyper_parameters['tile_embedder_pretrained'],
                                                    hyper_parameters['sparse_conv_n_channels_conv1'],
                                                    hyper_parameters['sparse_conv_n_channels_conv2'],
                                                    3, 3, hyper_parameters['sparse_map_downsample'],
                                                    hyper_parameters['wsi_embedding_classifier_n_inner_neurons'],
                                                    n_classes)
    sparseconvmil_model = torch.nn.DataParallel(sparseconvmil_model)
    optimizer = torch.optim.Adam(sparseconvmil_model.parameters(), hyper_parameters['lr'],
                                 weight_decay=hyper_parameters['reg'])
    loss_function = torch.nn.CrossEntropyLoss()

    # Loop through all epochs
    for epoch in range(hyper_parameters["epochs"]):
        loss, bac = perform_epoch(sparseconvmil_model, dataloader, optimizer, loss_function)
        print('Epoch', f'{epoch:3d}/{hyper_parameters["epochs"]}', f'    loss={loss:.3f}', f'    bac={bac:.3f}')


if __name__ == '__main__':
    main(_define_args())
