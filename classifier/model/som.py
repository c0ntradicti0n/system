from torch import nn


# Dummy data generation
def generate_dummy_sequences(batch_size=64, seq_len=5, embedding_dim=128, num_classes=9):
    # Generate random sequences of embeddings
    sequences = torch.randn(batch_size, seq_len, embedding_dim)
    # Generate random labels for each sequence
    labels = torch.randint(0, num_classes, (batch_size,))
    return sequences, labels


class Som(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=3, bidirectional=True, dropout_rate=0.2):
        super(Som, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=5, padding=1)
        self.relu = nn.ReLU()
        self.bidirectional = bidirectional
        # Adjusting the hidden dimension if using a bidirectional GRU, as the outputs will be concatenated
        final_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.gru = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,  # Dropout on all but the last layer
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(final_hidden_dim, output_dim)

    def forward(self, x):
        #x = x.transpose(1, 2)  # Conv1D expects (batch, channels, seq_len)
        #x = self.conv1d(x)
        #x = self.relu(x)
        #x = x.transpose(1, 2)  # Back to (batch, seq_len, channels) for GRU
        out, _ = self.gru(x)

        # If using a bidirectional GRU, out will contain concatenated hidden states from both directions
        # Here, we're only using the last hidden state(s) for classification
        if self.bidirectional:
            out = self.dropout(out[:, -1, :])  # Apply dropout to the concatenated final hidden state
        else:
            out = self.dropout(out[:, -1, :])  # Apply dropout to the final hidden state for unidirectional

        out = self.fc(out)
        return out


if __name__ == "__main__":
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from torch import nn, optim
    from sklearn.model_selection import train_test_split

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])  # Flatten the images
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split data into train and test
    train_data, test_data = train_test_split(mnist_data, test_size=0.2, random_state=42)


    def generate_sequences(data, sequence_length=5):
        sequences = []
        labels = []
        for _ in range(len(data) // sequence_length):
            indices = np.random.choice(len(data), sequence_length, replace=False)
            sequence = torch.stack([data[i][0] for i in indices])  # Get the flattened images
            label = sum(data[i][1] for i in indices)  # Sum of digits
            sequences.append(sequence)
            labels.append(label // 10)  # Example: classify based on the decade of the sum
        sequences_tensor = torch.stack(sequences)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return sequences_tensor, labels_tensor


    # Generate sequences for training and testing
    train_sequences, train_labels = generate_sequences(train_data, 5)
    test_sequences, test_labels = generate_sequences(test_data, 5)

    # DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model parameters
    input_size = 28 * 28  # Flattened MNIST images
    hidden_size = 128  # Hidden layer size
    output_size = 10  # Number of classes (based on the sum's decade)
    num_layers = 1  # Number of GRU layers

    # Parameters for the Transformer model
    input_size = 28 * 28  # Flattened MNIST images
    d_model = 512  # Size of the embedding
    output_size = 10  # Number of classes (based on the sum's decade)
    num_layers = 2  # Number of Transformer encoder layers
    nhead = 8  # Number of heads in the multiheadattention models
    dim_feedforward = 2048  # Size of the feedforward model in nn.TransformerEncoder
    dropout = 0.1  # Dropout rate

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and move it to the device
    model = TransformerModel(input_size, d_model, output_size, nhead, num_layers, dim_feedforward, dropout).to(device)

    # Remaining code for data loading and transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_data, test_data = train_test_split(mnist_data, test_size=0.2, random_state=42)

    # Generate sequences for training and testing
    # Ensure that you move your tensors to the device where necessary
    train_sequences, train_labels = generate_sequences(train_data,
                                                       5)  # Modify this function to move tensors to the device
    test_sequences, test_labels = generate_sequences(test_data, 5)  # Modify this function to move tensors to the device

    train_sequences, train_labels = train_sequences.to(device), train_labels.to(device)
    test_sequences, test_labels = test_sequences.to(device), test_labels.to(device)

    # DataLoader
    train_dataset = TensorDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

