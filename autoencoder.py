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


def trainAutoencoder(model, train_data, val_data, epochs, batchSize, lr, device='cpu'):
    """
    Train the autoencoder with validation data.
    :param model: Autoencoder model
    :param train_data: Training data (normalized)
    :param val_data: Validation data (normalized)
    :param epochs: Number of training epochs
    :param batchSize: Batch size for training
    :param lr: Learning rate
    :param device: Device ('cpu' or 'cuda')
    """
    # Prepare data
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
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

            batch = batch[0].to(device)
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

    torch.save(model.state_dict(), "autoencoder100-20")
    print("Model saved as autoencoder100-20")
