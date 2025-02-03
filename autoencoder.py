import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Autoencoder(nn.Module):
    def __init__(self, dimensions):
        """
        Initialize the autoencoder model.
        :param dimensions: a list with the dimensions of each layer
        """
        super(Autoencoder, self).__init__()

        if len(dimensions) < 2:
            raise ValueError('The dimensions list should contain at least 2 dimensions!')

        encoder_layers = []
        for i in range(len(dimensions) - 1):
            layer = nn.Linear(dimensions[i], dimensions[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            encoder_layers.append(layer)
            encoder_layers.append(nn.Tanh())

        # Decoder
        decoder_layers = []
        for i in range(len(dimensions) - 1, 0, -1):
            layer = nn.Linear(dimensions[i], dimensions[i - 1])
            nn.init.xavier_uniform_(layer.weight)
            decoder_layers.append(layer)
            decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the encoded data for the input tensor.

        :param x: Input tensor of shape (batch_size, inputSize)
        :return: Encoded representations as a tensor of shape (batch_size, bottleneckSize)
        """

        with torch.no_grad():
            return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)


class SlidingWindowDataset(Dataset):
    def __init__(self, data: torch.Tensor, window: int, step: int):
        self.data = data
        self.window = window
        self.step = step

    def __getitem__(self, index: int) -> torch.Tensor:
        index *= self.step

        if index + self.window > len(self.data):
            raise IndexError("Index out of range")

        return self.data[index:index+self.window]

    def __len__(self) -> int:
        maxStartIndex = len(self.data) - self.window

        return max(0, maxStartIndex // self.step + 1)


def trainAutoencoder(model: Autoencoder, trainCandles: np.ndarray, validCandles: np.ndarray, candlesNum: int, candleFeaturesNum: int, epochs=100, batchSize=32, lr=0.001, device='cpu') -> None:
    """
    Train the autoencoder with validation data.

    :param model: Autoencoder model
    :param trainCandles: Training data (normalized candles)
    :param validCandles: Validation data (normalized candles)
    :param candlesNum: Number of candles to use
    :param candleFeaturesNum: Number of features of each candle
    :param epochs: Number of training epochs
    :param batchSize: Batch size for training
    :param lr: Learning rate
    :param device: Device ('cpu' or 'cuda')
    """
    # Prepare data
    trainData = torch.from_numpy(trainCandles).reshape(-1)
    validData = torch.from_numpy(validCandles).reshape(-1)

    # create the datasets
    windowSize = candlesNum * candleFeaturesNum
    train_dataset = SlidingWindowDataset(trainData, windowSize, candleFeaturesNum)
    val_dataset = SlidingWindowDataset(validData, windowSize, candleFeaturesNum)

    # TODO check the impact of shuffling
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=1.0e-3)   # TODO tune the beta value

    # Training loop
    smallestError = float('inf')
    patience = 2  # Number of epochs to wait after last improvement
    wait = 0  # Counter for epochs without improvement
    best_model_state = None  # To store the best model state

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        print(f"Epoch {epoch + 1} / {epochs}")

        for i, batch in enumerate(train_loader):
            print(f"\rTraining batch {i + 1} / {len(train_loader)}", end="")

            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                total_val_loss += loss.item()

        # Logging losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.5e}, Validation Loss: {avg_val_loss:.5e}\n")

        # Early stopping logic
        if avg_val_loss < smallestError:
            smallestError = avg_val_loss
            best_model_state = model.state_dict()  # Save the best model state
            torch.save(best_model_state, "autoencoder.tmp")
            wait = 0  # Reset the wait counter
        else:
            wait += 1  # Increment the wait counter
            if wait >= patience:
                print(f"Early stopping! No improvement for {patience} epochs.")
                break

    # Restore the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored the best model based on validation loss.")

    print("Model saved as autoencoder.tmp")
