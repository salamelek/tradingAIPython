import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


class Autoencoder(nn.Module):
    def __init__(self, inputSize, bottleneckSize):
        """
        Initialize the autoencoder model.
        :param inputSize: Number of input features (e.g., 4 features * 100 candles = 400)
        :param bottleneckSize: Size of the bottleneck (compressed representation)
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(inputSize, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, bottleneckSize),
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneckSize, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, inputSize),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the encoded data for the input tensor.

        :param x: Input tensor of shape (batch_size, inputSize)
        :return: Encoded representations as a tensor of shape (batch_size, bottleneckSize)
        """

        with torch.no_grad():
            return self.encoder(x)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, step):
        self.data = data
        self.window = window
        self.step = step

    def __getitem__(self, index):
        index *= self.step
        return self.data[index:index+self.window]

    def __len__(self):
        return (len(self.data) - self.window) // self.step + 1


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

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.to(device)
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        print(f"Epoch {epoch + 1} / {epochs}")

        for i, batch in enumerate(train_loader):
            print(f"\rTraining batch {i + 1} / {len(train_loader)}", end="")

            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
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
                batch = batch[0].to(device)
                _, reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                total_val_loss += loss.item()

        # Logging losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.5e}, Validation Loss: {avg_val_loss:.5e}\n")

    torch.save(model.state_dict(), "autoencoder.tmp")
    print("Model saved as autoencoder.tmp")
