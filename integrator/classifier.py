import torch
import torch.nn as nn
from torch.nn import functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SimpleSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
        attention_weights = self.dropout(attention_weights)

        return attention_weights @ V


class BaseNetwork(nn.Module):
    """
    Basic network architecture that processes individual embeddings.
    """

    def __init__(self, input_dim, n_classes):
        super(BaseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1000),
            Mish(),
            nn.BatchNorm1d(1000),
            SimpleSelfAttention(1000),
            SimpleSelfAttention(1000),
            SimpleSelfAttention(1000),  # Stacking three layers of self-attention
            nn.Linear(1000, 500),
            Mish(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 250),
            nn.GELU(),
            nn.BatchNorm1d(250),
            nn.Linear(250, 125),
            nn.GELU(),
            nn.BatchNorm1d(125),
            nn.Linear(125, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_classes),  # Output layer for class logits
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        return self.fc(x)


class SiameseNetwork(nn.Module):
    """
    Siamese network for handling multiple embeddings using the base network.
    """

    def __init__(self, input_dim, n_classes):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork(input_dim, n_classes)

    def forward(self, *inputs):
        outputs = [self.base_network(input) for input in inputs]
        return torch.stack(outputs, dim=1)  # Convert list of tensors to a single tensor


def triplet_loss(anchor, positive, negative, margin=0.5):
    """
    Standard triplet loss function.
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def compute_loss(outputs):
    """
    Compute the triplet loss over the outputs of the Siamese network.
    It compares every embedding against every other.
    """
    total_loss = 0.0
    n = len(outputs)

    for i in range(n):
        for j in range(n):
            if i == j:  # Skip the same embedding
                continue

            for k in range(n):
                if k == i or k == j:  # Skip already used embeddings
                    continue

                total_loss += triplet_loss(outputs[i], outputs[j], outputs[k])

    return total_loss / (n * (n - 1) * (n - 2))  # Divided by total number of triplets


if __name__ == "__main__":
    input_dim = 128
    n_classes = 4
    config = {"n_classes": n_classes}
    siamese_network = SiameseNetwork(input_dim, config["n_classes"])

    sample = torch.rand((4, 4, 128))  # Example input
    outputs = siamese_network(*[sample[:, i, :] for i in range(sample.shape[1])])

    # Reshape outputs and labels
    outputs_reshaped = outputs.view(-1, config["n_classes"])

    train_labels = torch.randint(0, 4, (4, 4))  # Example labels tensor
    train_labels_reshaped = train_labels.view(-1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs_reshaped, train_labels_reshaped)
    print(loss)
