import sys
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

import config


def get_transform(split: str) -> transforms.Compose:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(config.IMAGE_SIZE),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    # Augment the train set with random horizontal flips.
    if split == 'train':
        transform = transforms.Compose(
            [transform,
             transforms.RandomHorizontalFlip()])
    return transform


def get_dataset(split: str) -> datasets.folder.ImageFolder:
    transform = get_transform(split)
    dataset = datasets.Country211(root=config.DATASET_DIRECTORY, split=split,
                                  transform=transform, download=True)
    return dataset


def get_data_loader(dataset: datasets.folder.ImageFolder, split: str) \
        -> DataLoader:
    shuffle = True if split == 'train' else False
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                             shuffle=shuffle, num_workers=config.NUM_WORKERS,
                             pin_memory=True)
    return data_loader


def get_model(class_count: int) -> nn.Module:
    # Get the pretrained EfficientNet model.
    model = getattr(models, config.MODEL_NAME)(pretrained=True)
    model.requires_grad_(False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                     class_count)
    return model


def run_epoch(mode: str, device, data_loader, model, criterion,
              optimizer=None) -> tuple[float, float, float]:
    model.train(mode == 'train')
    total_loss = 0
    total_correct = 0
    total_in_top_5 = 0
    with torch.set_grad_enabled(mode == 'train'):
        # Set file to sys.stdout for better coordination with print().
        batches = tqdm(data_loader, file=sys.stdout)
        batches.set_description(mode.capitalize())
        for batch in batches:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            if mode == 'train':
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if mode == 'train':
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += torch.sum(predictions == labels).item()
            # topk() returns a tuple of (values, indices).
            top_5_predictions = outputs.topk(5, dim=1)[1]
            total_in_top_5 += torch.sum(
                top_5_predictions == labels.unsqueeze(dim=1)).item()
    batch_count = len(data_loader)
    sample_count = len(data_loader.dataset)
    epoch_loss = total_loss / batch_count
    epoch_accuracy = total_correct / sample_count
    epoch_top_5_accuracy = total_in_top_5 / sample_count
    return epoch_loss, epoch_accuracy, epoch_top_5_accuracy


def log_epoch_results(epoch, writer, train_loss, train_accuracy,
                      train_top_5_accuracy, validation_loss,
                      validation_accuracy, validation_top_5_accuracy):
    print(f'Train: loss: {train_loss:.4f}, '
          f'accuracy: {train_accuracy:.2%}, '
          f'top-5 accuracy: {train_top_5_accuracy:.2%}\n'
          f'Validation: loss: {validation_loss:.4f}, '
          f'accuracy: {validation_accuracy:.2%}, '
          f'top-5 accuracy: {validation_top_5_accuracy:.2%}')
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Top-5 accuracy/train', train_top_5_accuracy, epoch)
    writer.add_scalar('Loss/validation', validation_loss, epoch)
    writer.add_scalar('Accuracy/validation', validation_accuracy,
                      epoch)
    writer.add_scalar('Top-5 accuracy/validation',
                      validation_top_5_accuracy, epoch)
    writer.flush()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    train_set = get_dataset('train')
    validation_set = get_dataset('valid')
    train_loader = get_data_loader(train_set, 'train')
    validation_loader = get_data_loader(validation_set, 'valid')
    class_count = len(train_set.classes)
    model = get_model(class_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    run_name = config.RUN_NAME or datetime.now().strftime("%y%m%d%H%M%S")
    run_directory = f'{config.RUNS_DIRECTORY}/{run_name}'
    writer = SummaryWriter(log_dir=run_directory)
    # Variables for early stopping. Stop training if the validation loss
    # does not improve for EARLY_STOPPING_PATIENCE epochs.
    best_model_state_dict = None
    best_model_loss = float('inf')
    best_model_accuracy = None
    best_model_top_5_accuracy = None
    epochs_since_improvement = 0
    for epoch in range(1, config.MAX_EPOCH_COUNT + 1):
        print(f'\nEpoch {epoch}/{config.MAX_EPOCH_COUNT}')
        train_loss, train_accuracy, train_top_5_accuracy = run_epoch(
            'train', device, train_loader, model, criterion, optimizer)
        validation_loss, validation_accuracy, validation_top_5_accuracy = (
            run_epoch('validation', device, validation_loader, model,
                      criterion))
        log_epoch_results(epoch, writer, train_loss, train_accuracy,
                          train_top_5_accuracy, validation_loss,
                          validation_accuracy, validation_top_5_accuracy)
        # Check for early stopping.
        if validation_loss < best_model_loss:
            best_model_state_dict = deepcopy(model.state_dict())
            best_model_loss = validation_loss
            best_model_accuracy = validation_accuracy
            best_model_top_5_accuracy = validation_top_5_accuracy
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.EARLY_STOPPING_PATIENCE:
                break
    # Save the hyperparameters.
    writer.add_hparams(
        hparam_dict={'model': config.MODEL_NAME,
                     'learning_rate': config.LEARNING_RATE,
                     'batch_size': config.BATCH_SIZE,
                     'optimizer': optimizer.__class__.__name__},
        metric_dict={'validation_loss': best_model_loss,
                     'validation_accuracy': best_model_accuracy,
                     'validation_top_5_accuracy': best_model_top_5_accuracy},
        run_name='hparams')
    writer.close()
    # Save the best model.
    torch.save(best_model_state_dict, f'{run_directory}/state_dict.pt')


if __name__ == '__main__':
    train()
