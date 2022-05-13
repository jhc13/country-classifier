import sys
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

import config
from datasets import get_data_loader, get_dataset


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        print(f'Device: {self.device}')
        splits = ('train', 'valid')
        self.datasets = {split: get_dataset(split) for split in splits}
        self.data_loaders = {split: get_data_loader(
            self.datasets[split], split) for split in splits}
        self.sample_count = {split: len(self.datasets[split]) for split
                             in splits}
        self.batch_count = {split: len(self.data_loaders[split]) for split
                            in splits}
        self.class_count = len(self.datasets['train'].classes)
        self.model = self.get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config.LEARNING_RATE)

    def get_model(self) -> nn.Module:
        model = getattr(models, config.MODEL_NAME)(pretrained=True)
        model.requires_grad_(False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         self.class_count)
        return model

    def run_epoch(self, mode: str) -> tuple[float, float]:
        total_loss = 0
        total_correct = 0
        with torch.set_grad_enabled(mode == 'train'):
            # Set file to sys.stdout for better coordination with print().
            batches = tqdm(self.data_loaders[mode], file=sys.stdout)
            batches.set_description(mode.capitalize())
            for batch in batches:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if mode == 'train':
                    self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                total_correct += torch.sum(predictions == labels).item()
        epoch_loss = total_loss / self.batch_count[mode]
        epoch_accuracy = total_correct / self.sample_count[mode]
        return epoch_loss, epoch_accuracy

    def run_train_epoch(self) -> tuple[float, float]:
        self.model.train()
        return self.run_epoch('train')

    def run_validation_epoch(self) -> tuple[float, float]:
        self.model.eval()
        return self.run_epoch('valid')

    def train(self):
        run_name = config.RUN_NAME or datetime.now().strftime("%y%m%d%H%M%S")
        run_directory = f'{config.RUNS_DIRECTORY}/{run_name}'
        writer = SummaryWriter(log_dir=run_directory)
        # Variables for early stopping. Stop training if the validation loss
        # does not improve for EARLY_STOPPING_PATIENCE epochs.
        best_model_state_dict = None
        best_model_loss = float('inf')
        best_model_accuracy = None
        epochs_since_improvement = 0
        for epoch in range(1, config.MAX_EPOCH_COUNT + 1):
            print(f'\nEpoch {epoch}/{config.MAX_EPOCH_COUNT}')
            train_loss, train_accuracy = self.run_train_epoch()
            validation_loss, validation_accuracy = self.run_validation_epoch()

            print(f'Train loss: {train_loss:.4f}, '
                  f'Train accuracy: {train_accuracy:.2%}\n'
                  f'Validation loss: {validation_loss:.4f}, '
                  f'Validation accuracy: {validation_accuracy:.2%}')
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/validation', validation_loss, epoch)
            writer.add_scalar('Accuracy/validation', validation_accuracy,
                              epoch)
            writer.flush()

            if validation_loss < best_model_loss:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_model_loss = validation_loss
                best_model_accuracy = validation_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= config.EARLY_STOPPING_PATIENCE:
                    break
        writer.add_hparams(
            hparam_dict={'model': config.MODEL_NAME,
                         'learning_rate': config.LEARNING_RATE,
                         'batch_size': config.BATCH_SIZE,
                         'optimizer': self.optimizer.__class__.__name__},
            metric_dict={'validation_loss': best_model_loss,
                         'validation_accuracy': best_model_accuracy},
            run_name='hparams')
        writer.close()
        torch.save(best_model_state_dict, f'{run_directory}/state_dict.pt')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
