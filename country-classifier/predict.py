import torch
import torch.nn.functional as F
from PIL import Image
from pycountry import countries
from torchvision import datasets

import config
from train import get_model, get_transform


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
    probabilities = F.softmax(output, dim=1)
    top_5_probabilities, top_5_labels = probabilities.topk(5, dim=1)
    top_5_probabilities = top_5_probabilities.squeeze()
    top_5_labels = top_5_labels.squeeze()
    top_5_countries = [countries.get(alpha_2=country_codes[label]).name
                       for label in top_5_labels]
    image.show()
    for country, probability in zip(top_5_countries, top_5_probabilities):
        print(f'{country}: {probability:.2%}')


if __name__ == '__main__':
    predict()
