# Country Classifier

A convolutional neural network (CNN) model for determining the country a photo
was taken in.

## Usage

The `country-classifier` directory contains 4 files:

- `config.py`
- `train.py`
- `test.py`
- `predict.py`

To train a model, set the configuration variables in `config.py` and run
`train.py`. The training run data and the best model are saved in the
provided directory (`runs` by default).

The trained model can be evaluated on the test set by providing the path to the
saved `state_dict.pt` file in `config.py` and running `test.py`.

Run `predict.py` after setting the relevant variables in `config.py` to make
predictions with the model. The model outputs the top `k` (5 by default) most
likely countries for the given image, as well as a
[choropleth map](https://en.wikipedia.org/wiki/Choropleth_map) of the
predicted probabilities for each country. The shapefiles used to generate the
map, from [Natural Earth](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/),
are located in the `shapefiles` directory.

## Dataset

The [Country211](https://github.com/openai/CLIP/blob/main/data/country211.md)
dataset from OpenAI was used to train the model. The dataset contains 63,300
images from 211 countries, with 150 train images, 50 validation images, and
100 test images for each country. The provided splits were used.

## Model

The pretrained [EfficientNet](https://arxiv.org/abs/1905.11946) models were
used as a base for transfer learning. Training a model from scratch was 
deemed infeasible, especially considering the small size of the dataset. 8
variations of EfficientNet models are available, increasing in size from
EfficientNet-B0 to EfficientNet-B7. The largest model that could be trained
without running out of memory on the available RTX 3080 10 GB GPU was
EfficientNet-B2, so this was chosen for the final model.

## Training

Several transfer learning approaches were tried, including:

- Training all layers with a uniform learning rate
- Training all layers, but using different learning rates for the `features`
  and `classifier` parts of the model
- Training only the `classifier` part of the model and freezing the other
  layers

The first approach produced the best results.

Early stopping was used to prevent overfitting, and the train set was augmented
with horizontal flips. Additional augmentations were also tried but did not
seem to improve the model's performance.

The `state_dict.pt` file for the final model can be found in the `models`
directory.
