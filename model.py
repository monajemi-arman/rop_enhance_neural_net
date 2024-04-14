#!/usr/bin/env python3
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import json
import pickle
import torch

# Change these
model_output_json = 'model_output.json'
model_input_pkl = 'model_input.pkl'
# Change in case any new parameter is added
image_parameters = {'contrast': [], 'brightness': [], 'gamma': [], 'cliplimit': [], 'strength': [],
                    'saturation': [], 'mode_intensity': [], 'mode_frequency': [], 'frequency_average': [],
                    'frequency_std_dev': [], 'intensity_average': [], 'standard_deviation': [],
                    'contrast_index': []
                    }

# Model Hyperparameters
batch_size = 1


class CustomDataset(Dataset):
    # Here, model output refers to the values the model should return (ground truth).
    # Model input (imported from pickle object) is the image stats the model sees as input.
    def __init__(self, model_output_json=model_output_json, model_input_pkl=model_input_pkl,
                 image_parameters=image_parameters, channel='green'):
        # Which color channel to train on, red, green, or blue
        self.channel = channel
        # Load data from disk
        with open(model_output_json, 'r') as fh:
            self.model_output = json.load(fh)
        with open(model_input_pkl, 'rb') as fh:
            self.model_input = pickle.load(fh)

        # Calculate min / max for later normalization
        for output in self.model_output:
            for key, value in output.items():
                if key != 'image_filename':
                    image_parameters[key].append(value)
        for input_ in self.model_input:
            for key, value in input_[self.channel].items():
                image_parameters[key].append(value)
        for key, value in image_parameters.items():
            image_parameters[key] = [min(value), max(value)]
        # Save for later normalization
        self.image_parameters = image_parameters

    def __len__(self):
        return len(self.model_output)

    def __getitem__(self, idx):
        # Normalize inputs and targets
        targets = self.normalize(self.model_output[idx])
        inputs = self.normalize(self.model_input[idx])
        return inputs, targets
    
    def normalize(self, the_dict, to_numpy=False):
        # If inputs are passed, separate the channel (green by default).
        if self.channel in the_dict.keys():
            the_dict = the_dict[self.channel]
        # Normalize
        normalized = []
        for key, value in the_dict.items():
            if key != 'image_filename':
                x_min, x_max = self.image_parameters[key]
                value = (value - x_min) / (x_max - x_min)
                normalized.append(value)
        if to_numpy:
            normalized = np.asarray(normalized, dtype=np.float32)
        else:
            normalized = torch.tensor(normalized, dtype=torch.float32)
        return normalized

    def denormalize(self, the_list):
        the_dict = {}
        # Rebuild the dict with names, so it's clear what each returned prediction is
        for key, value in zip(self.image_parameters.keys(), the_list):
            the_dict[key] = value
        # De-normalize values based on previous min and max
        for key, value in the_dict.items():
            x_min, x_max = self.image_parameters[key]
            value = value * (x_max - x_min) + x_min
            the_dict[key] = value
        return the_dict


class Model(pl.LightningModule):
    def __init__(self, in_features=7, out_features=6, hidden_features=30, hidden_layers=1, lr=1e-4):
        super().__init__()

        if hidden_layers < 2:
            self.model = nn.Sequential(nn.Linear(in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, out_features)
                                       )
        else:
            sequential_list = [nn.Linear(in_features, hidden_features), nn.ReLU()]
            for _ in range(hidden_layers):
                sequential_list.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU()])
            sequential_list.extend([nn.Linear(hidden_features, out_features)])
            self.model = nn.Sequential(*sequential_list)

        self.loss = nn.functional.mse_loss
        self.lr = lr

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = self.loss(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = self.loss(z, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = self.loss(z, y)
        self.log('test_loss', loss)
        return loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # Data
    dataset = CustomDataset()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train or Not
    if input("Do train? (y/N): ").lower() == 'y':
        # Testing out different combinations of layers
        combination_of_layers = [[1, 30], [2, 30], [5, 30], [1, 150], [2, 150], [5, 150], [7, 200]]
        for hidden_layers, hidden_features in combination_of_layers:
            # Logger
            logger = pl.loggers.TensorBoardLogger(
                save_dir='.',
                version=str(hidden_layers) + '-' + str(hidden_features),
                name='lightning_logs'
            )
            # Model
            model = Model(in_features=7, out_features=6, hidden_layers=hidden_layers, hidden_features=hidden_features)
            # Trainer
            trainer = pl.Trainer(logger=logger, max_epochs=3000)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Test the model
            trainer.test(model=model, dataloaders=test_loader)
    else:
        choice = input('Enter hidden_layers-hidden_features (e.g.: 2-150): ').split('-')
        choice = [int(x) for x in choice]
        ckpt_path = input('Enter checkpoint path: ')
        model = Model.load_from_checkpoint(ckpt_path, hidden_layers=choice[0], hidden_features=choice[1])
        print("Model loaded in variable 'model'.")
