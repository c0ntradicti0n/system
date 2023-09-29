import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from torch import nn, optim


def generate_samples(num_samples=1000, embedding_dim=128):
    means = [np.random.rand(embedding_dim) * i for i in range(4)]
    covs = [np.eye(embedding_dim) * 0.1 for _ in range(4)]
    labels = []
    samples = []

    for i in range(4):
        sample = multivariate_normal.rvs(
            mean=means[i], cov=covs[i], size=num_samples // 4
        )
        samples.extend(sample)
        labels.extend([i] * (num_samples // 4))

    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(
        np.array(labels), dtype=torch.long
    )


class SimpleSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.2)

        # Initialize the Linear layers
        nn.init.kaiming_normal_(self.query.weight)
        nn.init.kaiming_normal_(self.key.weight)
        nn.init.kaiming_normal_(self.value.weight)

        self.relation_network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )

        # Initialize the Linear layers in relation_network
        for layer in self.relation_network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
        attention_weights = self.dropout(attention_weights)

        return attention_weights @ V


class NTupleNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(NTupleNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, output_dim),
        )
        # Initialize the Linear layers in relation_network
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        # x = self.attention(x)  # Pass input through attention layer

        x = self.fc(x)
        return x


if __name__ == "__main__":
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 4
    n_samples = 4
    n_tuple_network = NTupleNetwork(embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(n_tuple_network.parameters(), lr=0.001)

    samples, labels = generate_samples()

    for epoch in range(100):
        optimizer.zero_grad()

        indices = np.random.choice(len(samples), size=n_samples, replace=False)
        selected_samples = torch.stack([samples[index] for index in indices])
        selected_labels = torch.tensor([labels[index] for index in indices])

        outputs = n_tuple_network(selected_samples)

        loss = criterion(outputs, selected_labels)
        loss.backward()

        optimizer.step()

        # Evaluation
        with torch.no_grad():
            outputs = n_tuple_network(samples)
            _, predicted = torch.max(outputs, 1)
            f1 = f1_score(labels, predicted, average="micro")
            print(f"Epoch {epoch + 1}, F1 Score: {f1:.4f}, Loss: {loss.item()}")
