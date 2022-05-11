import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

DATASET_DIRECTORY = '../dataset'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCH_COUNT = 10


def get_datasets() -> dict[str, datasets.folder.ImageFolder]:
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


def get_data_loaders(datasets_: dict[str, datasets.folder.ImageFolder]) \
        -> dict[str, DataLoader]:
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
        self.sample_count = {split: len(self.datasets[split]) for split
                             in self.datasets}
        self.batch_count = {split: len(self.data_loaders[split]) for split
                            in self.data_loaders}
        self.class_count = len(self.datasets['train'].classes)
        self.model = self.get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_model(self) -> nn.Module:
        model = models.efficientnet_b0(pretrained=True)
        model.requires_grad_(False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         self.class_count)
        return model

    def train_epoch(self) -> tuple[float, float]:
        total_loss = 0
        total_correct = 0
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
            predictions = outputs.argmax(dim=1)
            total_correct += torch.sum(predictions == labels).item()
        train_loss = total_loss / self.batch_count['train']
        train_accuracy = total_correct / self.sample_count['train']
        return train_loss, train_accuracy

    def train(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            train_loss, train_accuracy = self.train_epoch()
            print(f'Epoch {epoch}/{EPOCH_COUNT}: train loss: {train_loss:.4f}'
                  f', train accuracy: {train_accuracy:.2%}')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
