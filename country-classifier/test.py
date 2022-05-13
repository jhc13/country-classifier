import torch
from torch import nn

import config
from train import get_data_loader, get_dataset, get_model, run_epoch


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    dataset = get_dataset('test')
    data_loader = get_data_loader(dataset, 'test')
    class_count = len(dataset.classes)
    model = get_model(class_count).to(device)
    state_dict = torch.load(config.TEST_STATE_DICT_PATH)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = run_epoch('test', device, data_loader, model, criterion)
    print(f'Test loss: {loss:.4f}, test accuracy: {accuracy:.2%}')


if __name__ == '__main__':
    test()
