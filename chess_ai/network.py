import torch
import torch.nn as nn
import torch.optim as optim


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 512)  # Assuming a 64-square board and 12 possible piece types for each square
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 4672)  # Roughly the max number of possible moves in any position

        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output_activation(self.fc4(x))
        return x


# Initialize the network
net = ChessNet()

# Example input (random for demonstration; replace with your data)
# This should be a flattened bitboard representation of a chess position
example_input = torch.rand(1, 64 * 12)  # Batch size of 1 for simplicity

# Forward pass to get the move probabilities
move_probabilities = net(example_input)

print(move_probabilities)
