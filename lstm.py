import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CryptoLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Activation function (tanh to output values between -1 and 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        # Apply tanh activation to get output between -1 and 1
        out = self.tanh(out)

        return out


# Custom Dataset Class
class CryptoDatasetNextCandle(Dataset):
    def __init__(self, data: list[torch.Tensor], sequence_length: int):
        """
        data: List of sequences, where each sequence corresponds to a crypto pair.
              Each sequence is a tensor of shape (sequence_length, num_features).
        sequence_length: Length of each input sequence.
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = sequence[:-1]  # Input sequence (all candles except the last)
        y = sequence[-1]   # Target (last candle's market direction)
        return x, y


class TpSlDataset(Dataset):
    def __init__(self, candles: torch.Tensor, window: int):
        """
        candles: a tensor of ohlc data
        [
        [o1, h1, l1, c1],
        [o2, h2, l2, c2],
        [o3, h3, l3, c3],
        ...
        ]
        window: length of time sequence
        """

        self.candles = candles
        self.window = window

    def __len__(self):
        return len(self.candles) - self.window + 1

    def __getitem__(self, idx):
        """
        x: data window starting at index, normalised
        tanh([
        [h1/o1-1, l1/o1-1, c1/o1-1],
        [h2/o2-1, l2/o2-1, c2/o2-1],
        ...
        ])
        
        y: long or short
        """

        return x, y


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in dataloader:
            # Move data to device (e.g., GPU)
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            output = model(x)
            loss = criterion(output, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")


# Training
if __name__ == "__main__":
    # Hyperparameters
    input_size = 4  # Number of features (e.g., open, high, low, close)
    hidden_size = 64  # Number of hidden units in LSTM
    num_layers = 2  # Number of LSTM layers
    output_size = 1  # Output is a single value between -1 and 1
    sequence_length = 10  # Length of the input sequence (e.g., past 10 days)
    batch_size = 32  # Batch size - how many candles to look at
    num_epochs = 10

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryptoLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # Create dataset and dataloader
    # Example: data is a list of sequences, where each sequence is a tensor of shape (sequence_length, num_features)
    data = [torch.randn(sequence_length, input_size) for _ in range(1000)]  # Replace with real data
    dataset = CryptoDataset(data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()  # Use MSE for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)
