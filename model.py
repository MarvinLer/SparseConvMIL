__author__ = 'marvinler'

import math
import random

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet_factory

import sparseconvnet


class SparseConvMIL(nn.Module):
    def __init__(self, tile_embedder: nn.Module, sparse_cnn: nn.Module, wsi_embedding_classifier: nn.Module,
                 sparse_map_downsample: int, tile_coordinates_rotation_augmentation: bool,
                 tile_coordinates_flips_augmentation: bool, tile_coordinates_resize_augmentation: bool):
        super().__init__()
        self.tile_embedder = tile_embedder
        self.sparse_cnn = sparse_cnn
        self.wsi_embedding_classifier = wsi_embedding_classifier

        self.sparse_map_downsample = sparse_map_downsample

        # Data augmentation on tiles coordinates
        self.tile_coordinates_rotation_augmentation = tile_coordinates_rotation_augmentation
        self.tile_coordinates_flips_augmentation = tile_coordinates_flips_augmentation
        self.tile_coordinates_resize_augmentation = tile_coordinates_resize_augmentation

    def compute_tile_embeddings(self, tiles):
        """
        Computes concurrent and independent tile embedding with the tile embedder.
        :param tiles: tensor of tiles of expected shape (B_wsi, B_tiles, channels, width, height) with B_wsi equal to
            the number of considered WSI, and B_tiles equal to the number of tiles per considered WSI
        :return: a tensor of tiles embeddings of shape (B_wsi, B_tiles, latent_size)
        """
        # Flatten all tiles across all WSI:
        # (B_wsi, B_tiles, channels, width, height) -> (B_wsi*B_tiles, channels, width, height)
        tiles = tiles.view(tiles.shape[0] * tiles.shape[1], *tiles.shape[2:])
        return self.tile_embedder(tiles)

    @staticmethod
    def post_process_tiles_locations(tiles_locations):
        """
        Reformat the tiles locations into the proper expected format: the sparse-input CNN library sparseconvnet
            expects locations in the format
            [[tile1_loc_x, tile1_loc_y, batch_index_of_tile1],
             [tile2_loc_x, tile2_loc_y, batch_index_of_tile2],
             ...
             [tileN_loc_x, tileN_loc_y, batch_index_of_tileN]]
        :param tiles_locations: locations of sampled tiles with shape (B, n_tiles, 2) with B batch size, n_tiles the
            number of tiles per batch index and the other dimension for both coordinates_x and coordinates_y
        :return: a reformatted tensor of tiles locations with shape (n_tiles, 3)
        """
        device = tiles_locations.device
        reshaped_tiles_locations = tiles_locations.view(tiles_locations.shape[0]*tiles_locations.shape[1], -1)
        repeated_batch_indexes = torch.tensor([[b] for b in range(tiles_locations.shape[0])
                                               for _ in range(tiles_locations.shape[1])]).to(device)
        return torch.cat((reshaped_tiles_locations, repeated_batch_indexes), dim=1)

    def data_augment_tiles_locations(self, tiles_locations):
        """
        Perform data augmentation of the sparse map of tiles embeddings. First, a matrix of random rotations, flips,
            and resizes is instantiated. Then, a random translation vector is instantiated. The random translation is
            applied on the tiles coordinates, followed by the random rot+flips+resizes.
        :param tiles_locations: matrix of shape (batch_size, n_tiles_per_batch, 2) with tiles coordinates
        :return:
        """
        device = tiles_locations.device

        transform_matrix = torch.eye(2)
        # Random rotations
        if self.tile_coordinates_rotation_augmentation:
            theta = random.uniform(-180., 180.)
            rot_matrix = torch.tensor([[math.cos(theta), -math.sin(theta)],
                                       [math.sin(theta), math.cos(theta)]])
            transform_matrix = rot_matrix
        # Random flips
        if self.tile_coordinates_flips_augmentation:
            flip_h = random.choice([-1., 1.])
            flip_v = random.choice([-1., 1.])
            flip_matrix = torch.tensor([[flip_h, 0.],
                                        [0., flip_v]])
            transform_matrix = torch.mm(transform_matrix, flip_matrix)
        # Random resizes per axis
        if self.tile_coordinates_resize_augmentation:
            size_factor_h = 0.6 * random.random() + 0.7
            size_factor_v = 0.6 * random.random() + 0.7
            resize_matrix = torch.tensor([[size_factor_h, 0.],
                                          [0., size_factor_v]])
            transform_matrix = torch.mm(transform_matrix, resize_matrix)

        # First random translates ids, then apply matrix
        effective_sizes = torch.max(tiles_locations, dim=0)[0] - torch.min(tiles_locations, dim=0)[0]
        random_indexes = [random.randint(0, int(size)) for size in effective_sizes]
        translation_matrix = torch.tensor(random_indexes)
        tiles_locations -= translation_matrix.to(device)
        # Applies transformation
        tiles_locations = torch.mm(tiles_locations.float(), transform_matrix.to(device)).long()

        # Offsets tiles to the leftmost and rightmost
        tiles_locations -= torch.min(tiles_locations, dim=0, keepdim=True)[0]
        return tiles_locations

    def forward(self, x, tiles_original_locations):
        tile_embeddings = self.compute_tile_embeddings(x)

        # Builds the sparse map: assign each embedding into its specified location within an empty sparse map
        # First applies downsampling to original tiles locations (see paper)
        tiles_locations = tiles_original_locations / self.sparse_map_downsample
        # Perform data augmentation of the tiles locations, i.e. spatial data augmentation of the sparse map
        tiles_locations = torch.stack([self.data_augment_tiles_locations(tl) for tl in tiles_locations], dim=0)
        tiles_locations = tiles_locations.to(x.device)
        # Converts tiles locations into the expected format for sparseconvnet
        tiles_locations = self.post_process_tiles_locations(tiles_locations)
        # Instantiates an empty sparse map container for sparseconvnet. Spatial_size is set to the maximum of tiles
        # locations for both axis; mode=4 implies that two embeddings at the same location are averaged elementwise
        input_layer = sparseconvnet.InputLayer(dimension=2,
                                               spatial_size=(int(torch.max(tiles_locations[:, 0])) + 1,
                                                             int(torch.max(tiles_locations[:, 1])) + 1),
                                               mode=4)
        # Assign each tile embedding to their corresponding post-processed tile location
        sparse_map = input_layer([tiles_locations, tile_embeddings])

        wsi_embedding = self.sparse_cnn(sparse_map)
        wsi_embedding = torch.flatten(wsi_embedding, start_dim=1)

        return self.wsi_embedding_classifier(wsi_embedding)


class SparseAdaptiveAvgPool(nn.AdaptiveAvgPool1d):
    """
    Custom pooling layer that transform a (c, w, h) input sparse tensor into a (c,) output sparse tensor
    """
    def __init__(self, output_size):
        super().__init__(output_size)

    def forward(self, sparse_tensor_input):
        input_features = sparse_tensor_input.features
        input_locations = sparse_tensor_input.get_spatial_locations()

        res = []
        for batch_idx in torch.unique(input_locations[..., 2]):
            pooled = super().forward(input_features[input_locations[..., 2] == batch_idx].transpose(0, 1).unsqueeze(0))
            res.append(pooled)

        return torch.cat(res, dim=0)


def get_classifier(input_n_neurons: int, inner_n_neurons: int, n_classes: int):
    """
    Instantiates a ReLU-activated 1-hidden layer MLP.
    :param input_n_neurons: vector size of input data (should be WSI embedding)
    :param inner_n_neurons: number of inner neurons
    :param n_classes: number of output classes
    :return: a Sequential model
    """
    return nn.Sequential(
        nn.Linear(input_n_neurons, inner_n_neurons),
        nn.ReLU(inplace=True),
        nn.Linear(inner_n_neurons, n_classes),
    )


def get_resnet_model(resnet_architecture: str, pretrained: bool):
    """
    Instantiates a ResNet architecture without the finale FC layer.
    :param resnet_architecture: the desired ResNet architecture (e.g. ResNet34 or Wide_Resnet50_2)
    :param pretrained: True to load an architecture pretrained on Imagenet, otherwise standard initialization
    :return: (a Sequential model, number of output channels from the returned model)
    """
    assert resnet_architecture.lower() in resnet_factory.__all__
    resnet_model = getattr(resnet_factory, resnet_architecture.lower())(pretrained, progress=True)
    n_output_channels = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential()
    return resnet_model, n_output_channels


def get_two_layers_sparse_cnn(input_n_channels: int, n_out_channels_conv1: int, n_out_channels_conv2: int,
                              filter_width_conv1: int, filter_width_conv2: int):
    """
    Instantiates a 2-layers sparse-input ReLU-activated CNN, with a GlobalAveragePooling to reduce spatial
        dimensions to 1.
    :param input_n_channels: vector size of input data (should be the size of each tile embedding)
    :param n_out_channels_conv1: number of output channels for the first convolution
    :param n_out_channels_conv2: number of output channels for the second convolution
    :param filter_width_conv1: width of conv filters for the first convolution
    :param filter_width_conv2: width of conv filters for the second convolution
    :return: a sparseconvnet Sequential model
    """
    return sparseconvnet.Sequential(
        sparseconvnet.SubmanifoldConvolution(2, input_n_channels, n_out_channels_conv1, filter_width_conv1, True),
        sparseconvnet.ReLU(),
        sparseconvnet.SubmanifoldConvolution(2, n_out_channels_conv1, n_out_channels_conv2, filter_width_conv2, True),
        sparseconvnet.ReLU(),
        SparseAdaptiveAvgPool(1),
    )


def instantiate_sparseconvmil(tile_embedder_resnet_architecture, tile_embedder_pretrained, n_out_channels_conv1,
                              n_out_channels_conv2, filter_width_conv1, filter_width_conv2, sparse_map_downsample,
                              wsi_classifier_input_n_neurons, n_classes):
    """
    Instantiates a complete SparseConvMIL model:
        1. build a tile embedder (ResNet)
        2. then a pooling function (2-layers sparse-input CNN)
        3. then a classifier (1-hidden layer MLP)
    :param tile_embedder_resnet_architecture: resnet architecture of the tile embedder
    :param tile_embedder_pretrained: True to instantiate an Imagenet-pretrained tile embedder
    :param n_out_channels_conv1: number of output channels for the first convolution of the sparse-input pooling
    :param n_out_channels_conv2: number of output channels for the second convolution of the sparse-input pooling
    :param filter_width_conv1: width of conv filters for the first convolution of the sparse-input pooling
    :param filter_width_conv2: width of conv filters for the second convolution of the sparse-input pooling
    :param sparse_map_downsample: downsampling factor applied to the location of the sparse map
    :param wsi_classifier_input_n_neurons: number of inner neurons of the WSI embedding classifier
    :param n_classes: number of output classes
    :return: a Sequential model
    """
    tile_embedder, n_output_channels_tile_embedding = get_resnet_model(tile_embedder_resnet_architecture,
                                                                       tile_embedder_pretrained)
    sparse_input_pooling = get_two_layers_sparse_cnn(n_output_channels_tile_embedding, n_out_channels_conv1,
                                                     n_out_channels_conv2, filter_width_conv1, filter_width_conv2)
    wsi_embedding_classifier = get_classifier(n_out_channels_conv2, wsi_classifier_input_n_neurons, n_classes)

    sparseconvmil_model = SparseConvMIL(tile_embedder, sparse_input_pooling, wsi_embedding_classifier,
                                        sparse_map_downsample, True, True, True)
    return sparseconvmil_model
