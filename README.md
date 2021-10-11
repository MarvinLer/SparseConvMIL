# SparseConvMIL
Library for ["Sparse Convolutional Context-Aware Multiple Instance Learning for Whole Slide Image 
Classification"](paper_SparseConvMIL.pdf) (Best paper at the MICCAI workshop on Computational Pathology 2021).

This repository contains the full kit for training SparseConvMIL with any type of ResNet architecture as
tile embedder.
The provided data is from [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/).

## Sparse Convolutional Multiple Instance Learning
SparseConvMIL is a powerful and generic multiple instance learning architecture specifically designed
to leverage spatial information in whole slide images.
This is done by building a sparse map that contains the embeddings of sampled tiles, which are
placed at the locations of the associated tiles within the source whole slide image.

This framework has shown state-of-the-art performance for subtype classification compared to 
conventional multiple instance learning approaches.

<p align="center">
    <img src="img/sparseconvmil_architecture.png" alt="Schematic representation of SparseConvMIL, with a 2-layers sparse-input CNN" width="600"/>
</p>

## Hello world

To run the demo version:

`python -m training`

More info about the hyper-parameters with:

`python -m training --help`

All models, including the tile embedder, the sparse-input pooling, the WSI embedding classifier and 
SparseConvMIL are located within the [model.py](model.py) file.
If you want to change one of several of these, check this file!

To accomodate for your data, either copy the architecture as displayed in the 
[sample_data](sample_data), or check the [dataset.py](dataset.py) file for expected data architecture 
or to change data loading.

## Setup

Clone this repo, create a virtual environment and download necessary packages:
```
git clone git@github.com:MarvinLer/SparseConvMIL
cd SparseConvMIL
virtualenv -p python3 venv; source venv/bin/activate
pip install -r requirements.txt
```

This library also relies on [SparseConvNet](https://github.com/facebookresearch/SparseConvNet):
```
# Get and install sparseconvnet
git clone git@github.com:facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash develop.sh
```

## Citations
If you find this code useful in your research then please cite:

["Sparse Convolutional Context-Aware Multiple Instance Learning for Whole Slide Image 
Classification, COMPAY 2021"](https://proceedings.mlr.press/v156/lerousseau21a.html).

```
@inproceedings{lerousseau2021sparseconvmil,
  title={SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance Learning for Whole Slide Image Classification},
  author={Lerousseau, Marvin and Vakalopoulou, Maria and Deutsch, Eric and Paragios, Nikos},
  booktitle={MICCAI Workshop on Computational Pathology},
  pages={129--139},
  year={2021},
  organization={PMLR}
}
```

```
Lerousseau, M., Vakalopoulou, M., Deutsch, E. and Paragios, N., 2021, September. 
SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance Learning for Whole Slide Image Classification. 
In MICCAI Workshop on Computational Pathology (pp. 129-139). PMLR.
```

## License
SparseConvMIL is GNU AGPLv3 licensed, as found in the LICENSE file.
