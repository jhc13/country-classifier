from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_DIRECTORY = '../dataset'


def get_dataset(split: str) -> datasets.folder.ImageFolder:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    if split == 'train':
        transform = transforms.Compose(
            [transform,
             transforms.RandomHorizontalFlip()])
    dataset = datasets.Country211(root=DATASET_DIRECTORY, split=split,
                                  transform=transform, download=True)
    return dataset


def get_data_loader(dataset: datasets.folder.ImageFolder, split: str,
                    batch_size: int, num_workers: int) \
        -> DataLoader:
    shuffle = True if split == 'train' else False
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=True)
    return data_loader
