import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    # TODO do i use ReLU?
    def __init__(self, inputSize, bottleneckSize):
        """
        Initialize the autoencoder model.
        :param inputSize: Number of input features (e.g., 4 features * 100 candles = 400)
        :param bottleneckSize: Size of the bottleneck (compressed representation)
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(inputSize, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneckSize),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneckSize, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, inputSize)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def trainAutoencoder(model, data, epochs=20, batchSize=32, lr=0.001, device='cpu'):
    """
    Train the autoencoder.
    :param model: Autoencoder model
    :param data: Input data (normalized)
    :param epochs: Number of training epochs
    :param batchSize: Batch size for training
    :param lr: Learning rate
    :param device: Device ('cpu' or 'cuda')
    """
    # Prepare data
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")
