#!/usr/bin/env python3
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import json
import pickle
import torch
import os

# Change these
model_output_json = 'model_output.json'
model_input_pkl = 'model_input.pkl'

# Define combinations of layers
combination_of_layers = [[2, 150]]

# Change in case any new parameter is added
image_parameters = {
    'contrast': [], 'brightness': [], 'gamma': [], 'cliplimit': [], 'strength': [],
    'saturation': [],
    'red_mode_intensity': [], 'red_mode_frequency': [], 'red_frequency_average': [],
    'red_frequency_std_dev': [], 'red_intensity_average': [], 'red_standard_deviation': [],
    'red_contrast_index': [],
    'green_mode_intensity': [], 'green_mode_frequency': [], 'green_frequency_average': [],
    'green_frequency_std_dev': [], 'green_intensity_average': [], 'green_standard_deviation': [],
    'green_contrast_index': [],
    'blue_mode_intensity': [], 'blue_mode_frequency': [], 'blue_frequency_average': [],
    'blue_frequency_std_dev': [], 'blue_intensity_average': [], 'blue_standard_deviation': [],
    'blue_contrast_index': []
}

# Model Hyperparameters
batch_size = 32

class CustomDataset(Dataset):
    def __init__(self, model_output_json=model_output_json, model_input_pkl=model_input_pkl,
                 image_parameters=image_parameters):
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
            for channel in ['red', 'green', 'blue']:
                for key, value in input_[channel].items():
                    image_parameters[f"{channel}_{key}"].append(value)
        for key, value in image_parameters.items():
            image_parameters[key] = [min(value), max(value)]
        # Save for later normalization
        self.image_parameters = image_parameters

    def __len__(self):
        return len(self.model_output)

    def __getitem__(self, idx):
        # Normalize inputs and targets
        targets = self.normalize_output(self.model_output[idx])
        inputs = self.normalize_input(self.model_input[idx])
        return inputs, targets

    def normalize_output(self, the_dict):
        normalized = []
        for key in ['contrast', 'brightness', 'gamma', 'cliplimit', 'strength', 'saturation']:
            x_min, x_max = self.image_parameters[key]
            value = the_dict.get(key, 0)
            if x_max > x_min:
                value = (value - x_min) / (x_max - x_min)
            normalized.append(value)
        return torch.tensor(normalized, dtype=torch.float32)

    def normalize_input(self, the_dict):
        normalized = []
        for channel in ['red', 'green', 'blue']:
            channel_dict = the_dict.get(channel, {})
            for key in ['mode_intensity', 'mode_frequency', 'frequency_average',
                        'frequency_std_dev', 'intensity_average', 'standard_deviation',
                        'contrast_index']:
                full_key = f"{channel}_{key}"
                value = channel_dict.get(key, 0)
                x_min, x_max = self.image_parameters.get(full_key, [0, 1])
                if x_max > x_min:
                    value = (value - x_min) / (x_max - x_min)
                normalized.append(value)
        return torch.tensor(normalized, dtype=torch.float32)

    def normalize_full(self, the_dict, to_numpy=False):
        """
        Normalize a given dictionary of image stats for all channels.

        Parameters:
        - the_dict: dict, image stats to normalize.
        - to_numpy: bool, if True return as numpy array.

        Returns:
        - Normalized tensor or numpy array with all channels.
        """
        normalized = []
        for channel in ['red', 'green', 'blue']:
            channel_dict = the_dict.get(channel, {})
            for key in ['mode_intensity', 'mode_frequency', 'frequency_average',
                        'frequency_std_dev', 'intensity_average', 'standard_deviation',
                        'contrast_index']:
                full_key = f"{channel}_{key}"
                value = channel_dict.get(key, 0)
                x_min, x_max = self.image_parameters.get(full_key, [0, 1])
                if x_max > x_min:
                    value = (value - x_min) / (x_max - x_min)
                normalized.append(value)
        if to_numpy:
            return np.array(normalized, dtype=np.float32)
        else:
            return torch.tensor(normalized, dtype=torch.float32)

    def denormalize(self, the_list):
        the_dict = {}
        # Rebuild the dict with names, so it's clear what each returned prediction is
        for key, value in zip(['contrast', 'brightness', 'gamma', 'cliplimit', 'strength', 'saturation'], the_list):
            x_min, x_max = self.image_parameters[key]
            value = value * (x_max - x_min) + x_min
            the_dict[key] = value
        return the_dict

class Model(pl.LightningModule):
    def __init__(self, in_features=21, out_features=6, hidden_features=150, hidden_layers=2, lr=1e-3, dropout=0.5, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_features, out_features))
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_fn(z, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_fn(z, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_fn(z, y)
        self.log('test_loss', loss)
        return loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def export_model_to_onnx(model, hidden_layers, hidden_features, in_features=21, onnx_dir='onnx_models'):
    """
    Exports the given model to an ONNX file.

    Parameters:
    - model: Trained PyTorch Lightning model.
    - hidden_layers: Number of hidden layers.
    - hidden_features: Number of hidden features.
    - in_features: Number of input features.
    - onnx_dir: Directory to save ONNX models.
    """
    os.makedirs(onnx_dir, exist_ok=True)
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, in_features)
    onnx_filename = os.path.join(
        onnx_dir, f"model_h{hidden_layers}_f{hidden_features}.onnx"
    )
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Model exported to {onnx_filename}")

if __name__ == '__main__':
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    # Data
    dataset = CustomDataset()
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Train or Not
    if input("Do train? (y/N): ").strip().lower() == 'y':
        for hidden_layers, hidden_features in combination_of_layers:
            print(f"\nTraining model with {hidden_layers} hidden layer(s) and {hidden_features} hidden feature(s).")

            logger = pl.loggers.TensorBoardLogger(
                save_dir='.',
                version=f"{hidden_layers}-{hidden_features}",
                name='lightning_logs'
            )

            model = Model(
                in_features=21,
                out_features=6,
                hidden_layers=hidden_layers,
                hidden_features=hidden_features,
                lr=1e-3,
                dropout=0.5,
                weight_decay=1e-5
            )

            trainer = pl.Trainer(
                logger=logger,
                max_epochs=2000,
                callbacks=[early_stop_callback, checkpoint_callback],
                gradient_clip_val=1.0
            )

            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            trainer.test(model=model, dataloaders=test_loader)
            export_model_to_onnx(model, hidden_layers, hidden_features)

    else:
        choice = input('Enter hidden_layers-hidden_features (e.g., 2-150): ').strip().split('-')
        if len(choice) != 2:
            print("Invalid input format. Please enter in 'hidden_layers-hidden_features' format (e.g., 2-150).")
            exit(1)
        try:
            choice = [int(x) for x in choice]
        except ValueError:
            print("Both hidden_layers and hidden_features should be integers.")
            exit(1)
        ckpt_path = input('Enter checkpoint path: ').strip()
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint file '{ckpt_path}' does not exist.")
            exit(1)
        model = Model.load_from_checkpoint(
            ckpt_path,
            hidden_layers=choice[0],
            hidden_features=choice[1],
            in_features=21,
            out_features=6,
            dropout=0.5,
            weight_decay=1e-5
        )
        print("Model loaded in variable 'model'.")

        export_choice = input("Do you want to export the loaded model to ONNX? (y/N): ").strip().lower()
        if export_choice == 'y':
            export_model_to_onnx(model, choice[0], choice[1])
