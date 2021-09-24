__author__ = 'marvinler'

import os
import random

import torch.utils.data.dataset
from PIL import Image
from torch.utils.data.dataset import T_co
from torchvision import transforms


def pil_loader(path):
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Dataset(torch.utils.data.dataset.Dataset):
    """
    Pre-fetch all tiles files from all slides, as well as slides labels.
    Expects a tile folder as described in doc of self.load_tile_paths.
    Also expects a CSV ground-truth file as described in self.load_labels.
    """
    def __init__(self, slides_parent_folder, slides_label_filepath, n_sampled_tiles_per_slide,
                 tile_augmentation=None):
        """
        Loads tiles paths, loads slide labels, instantiates data augmentation.
        :param slides_parent_folder: parent folder containing 1 folder per WSI (each WSI folder containing tiles as
            images)
        :param slides_label_filepath: CSV filepath containing the labels of slides
        :param n_sampled_tiles_per_slide: number of tiles to be sampled per sampled WSI
        :param tile_augmentation: None or a torchvision data augmentation transforms
        """
        self.slides_parent_folder = slides_parent_folder
        assert os.path.exists(slides_parent_folder)
        self.slides_label_filepath = slides_label_filepath
        assert os.path.exists(slides_label_filepath)

        self.n_sampled_tiles_per_slide = n_sampled_tiles_per_slide

        self.tile_augmentation = tile_augmentation if tile_augmentation else transforms.Compose([
            transforms.ToTensor()
        ])

        self.tiles_files, self.tiles_locations, self.slides_ids = self.load_tile_paths()
        self.slide_labels, self.correspondence_digit_label_name = self.load_labels()
        self.n_classes = len(self.correspondence_digit_label_name)
        # Check that every slide folder has a corresponding label
        assert all([slide_id in self.slide_labels.keys() for slide_id in self.slides_ids])

    def load_tile_paths(self) -> (list, list):
        """
        Seeks and store all tiles paths. Expects folder hierarchy e.g.:
            parent_folder
            |    slide_folder_1
            |    |    tile1
            |    |    tile2
            |    slide_folder_2
            |    |    tile1
            |    |    tile2
            Each tile should be named with their coordinates, e.g. 16_23.png for tile row 16 column 23 within the WSI.
        :return: (list of lists: each inner list contains all images from each (slide) folder in self.parent_folder,
                  associated tiles coordinates in a list of 2-elements lists e.g. [[16, 23], [2, 39]],
                  the list of slides ids i.e. the name of the slides folders)
        """
        parent_folder = self.slides_parent_folder
        slide_folders = list(map(lambda f: os.path.join(parent_folder, f), os.listdir(parent_folder)))
        slide_folders = list(filter(lambda f: not os.path.isfile(f), slide_folders))
        slide_folders_ids = list(map(os.path.basename, slide_folders))

        # Get absolute tiles files
        tiles_paths = list(map(lambda slide_folder:
                               list(map(lambda tile: os.path.join(slide_folder, tile),
                                        list(filter(lambda f: f.endswith(('.png', '.jpg', '.jpeg', '.bmp')),
                                                    os.listdir(slide_folder))))),
                               slide_folders))

        # Recover tiles coordinates
        tiles_locations = list(map(lambda tiles_files_per_slide:
                                   list(map(lambda tile_path:
                                            list(map(int, os.path.basename(tile_path).split('.')[0].split('_'))),
                                            tiles_files_per_slide)),
                                   tiles_paths))
        assert len(slide_folders_ids) == len(tiles_locations) == len(tiles_paths)
        return tiles_paths, tiles_locations, slide_folders_ids

    def load_labels(self):
        """
        Loads labels from a CSV file, then converts all labels to digits.
        Expects two columns: the first with slide folders names of each slide, the second with an associated label
        (any type of label).
        :return: (a dictionary of length the number of CSV entries with keys: slides ids and values: its corresponding
            digit, a dictionary of correspondence between digit and original label name)
        """
        csv_filepath = self.slides_label_filepath
        with open(csv_filepath, 'r') as f:
            content = f.read().splitlines()
        cells = [line.split(',') for line in content]
        slides_ids = list(map(lambda cell: cell[0], cells))
        slides_labels = list(map(lambda cell: cell[1], cells))

        # Converts labels to digits
        unique_labels = list(set(slides_labels))
        assert len(unique_labels) > 1, f'Expected at least two labels, found {len(unique_labels)}'
        label_to_digit = {label: digit for digit, label in enumerate(unique_labels)}
        digit_to_label = {digit: label for digit, label in enumerate(unique_labels)}
        slides_labels = {slide_id: label_to_digit[label] for slide_id, label in zip(slides_ids, slides_labels)}

        return slides_labels, digit_to_label

    def __getitem__(self, slide_index) -> T_co:
        slide_tiles_paths = self.tiles_files[slide_index]
        slide_tiles_locations = self.tiles_locations[slide_index]
        # Randomly samples tiles among all tiles of slide
        n_tiles = len(slide_tiles_paths)
        sampled_tiles_indexes = random.choices(range(n_tiles), k=self.n_sampled_tiles_per_slide)
        # Loads all tiles and their locations
        tiles = list(map(pil_loader, [slide_tiles_paths[i] for i in sampled_tiles_indexes]))
        tiles = torch.stack(list(map(self.tile_augmentation, tiles)))
        tiles_locations = torch.tensor([slide_tiles_locations[i] for i in sampled_tiles_indexes])

        # Load associated slide label
        slide_id = self.slides_ids[slide_index]
        slide_label = self.slide_labels[slide_id]

        return tiles, tiles_locations, slide_label, slide_id

    def __len__(self):
        # Length is set to the number of slides
        return len(self.tiles_files)
