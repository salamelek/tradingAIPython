from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
            nn.Linear(inputSize, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, bottleneckSize),
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneckSize, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, inputSize),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def trainAutoencoder(model, data, epochs, batchSize, lr, device='cpu'):
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
        pbar = tqdm(loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(Batch_Loss=loss.item())

        # Print final epoch loss after progress bar
        avg_loss = total_loss / len(loader)
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
