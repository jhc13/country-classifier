import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


DATASET_DIRECTORY = '../dataset'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCH_COUNT = 10


def get_datasets():
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    train_transform = transforms.Compose(
        [test_transform,
         transforms.RandomHorizontalFlip()])

    train_set = datasets.Country211(root=DATASET_DIRECTORY, split='train',
                                    transform=train_transform, download=True)
    validation_set = datasets.Country211(root=DATASET_DIRECTORY, split='valid',
                                         transform=test_transform,
                                         download=True)
    test_set = datasets.Country211(root=DATASET_DIRECTORY, split='test',
                                   transform=test_transform, download=True)
    return train_set, validation_set, test_set


def get_data_loaders(train_set, validation_set, test_set):
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)
    return train_loader, validation_loader, test_loader


def get_model(class_count):
    model = models.efficientnet_b0(pretrained=True)
    model.requires_grad_(False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                     class_count)
    return model


def main():
    train_set, validation_set, test_set = get_datasets()
    train_loader, validation_loader, test_loader = get_data_loaders(
        train_set, validation_set, test_set)
    class_count = len(train_set.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = get_model(class_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCH_COUNT + 1):
        total_loss = 0
        for batch in tqdm(train_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}/{EPOCH_COUNT}: '
              f'loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    main()
