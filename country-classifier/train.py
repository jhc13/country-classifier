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
    return {'train': train_set, 'validation': validation_set, 'test': test_set}


def get_data_loaders(datasets_):
    train_loader = DataLoader(datasets_['train'], batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = DataLoader(datasets_['validation'],
                                   batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=8, pin_memory=True)
    test_loader = DataLoader(datasets_['test'], batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    return {'train': train_loader, 'validation': validation_loader,
            'test': test_loader}


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        print(f'Device: {self.device}')
        self.datasets = get_datasets()
        self.data_loaders = get_data_loaders(self.datasets)
        self.class_count = len(self.datasets['train'].classes)
        self.model = self.get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_model(self):
        model = models.efficientnet_b0(pretrained=True)
        model.requires_grad_(False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         self.class_count)
        return model

    def train(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            total_loss = 0
            for batch in tqdm(self.data_loaders['train']):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(self.data_loaders['train'])
            print(f'Epoch {epoch}/{EPOCH_COUNT}: '
                  f'loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
