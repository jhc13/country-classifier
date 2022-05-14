import geopandas
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pycountry import countries
from torchvision import datasets

import config
from train import get_model, get_transform


def get_country_name(country_code: str) -> str:
    # pycountry does not include Kosovo.
    if country_code == 'XK':
        return 'Kosovo'
    return countries.get(alpha_2=country_code).name


def print_top_k_predictions(k: int, probabilities: torch.Tensor,
                            country_codes: list[str]):
    top_probabilities, top_labels = probabilities.topk(k)
    top_country_names = [get_country_name(country_codes[label]) for label
                         in top_labels]
    for i, (country_name, probability) in enumerate(
            zip(top_country_names, top_probabilities)):
        print(f'{i + 1}. {country_name}: {probability:.2%}')


def plot_map(probabilities: torch.Tensor, country_codes: list[str]):
    world_map = geopandas.read_file(config.MAP_SHAPEFILE_PATH)
    world_map['probability'] = 0
    for country_code, probability in zip(country_codes, probabilities):
        world_map.loc[world_map['ISO_A2_EH'] == country_code,
                      'probability'] = probability.item()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='5%', pad=0.1)
    world_map.plot(column='probability', cmap='viridis', legend=True, ax=ax,
                   cax=cax)
    ax.set_facecolor('#d3e3e9')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(tight=True)
    cax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    fig.tight_layout()
    fig.show()
    if config.PLOT_SAVE_PATH:
        fig.savefig(config.PLOT_SAVE_PATH)


def predict():
    dataset = datasets.Country211(root=config.DATASET_DIRECTORY)
    country_codes = dataset.classes
    class_count = len(country_codes)
    model = get_model(class_count)
    state_dict = torch.load(config.PREDICT_STATE_DICT_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    image = Image.open(config.PREDICT_IMAGE_PATH)
    transform = get_transform('test')
    transformed_image = transform(image).unsqueeze(dim=0)
    with torch.no_grad():
        output = model(transformed_image)
    probabilities = F.softmax(output, dim=1).squeeze()
    print_top_k_predictions(config.PREDICT_TOP_K, probabilities, country_codes)
    plot_map(probabilities, country_codes)


if __name__ == '__main__':
    predict()
