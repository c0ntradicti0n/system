import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from torch import nn


def generate_samples(num_samples=1000, embedding_dim=128):
    means = [np.random.rand(embedding_dim) * i for i in range(4)]
    covs = [np.eye(embedding_dim) * 0.1 for _ in range(4)]
    labels = []
    samples = []

    for i in range(4):
        sample = multivariate_normal.rvs(
            mean=means[i], cov=covs[i], size=num_samples * 4
        )
        samples.extend(sample)
        labels.extend([i] * (num_samples * 4))

    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(
        np.array(labels), dtype=torch.long
    )


class NTupleNetwork(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(NTupleNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 1),
            nn.GELU(),
            nn.Linear(embedding_dim * 1, int(embedding_dim * 0.5 // 1)),
            nn.GELU(),
            nn.Linear(int(embedding_dim * 0.5 // 1), int(embedding_dim * 0.25 // 1)),
            nn.GELU(),
            nn.Linear(int(embedding_dim * 0.25 // 1), int(embedding_dim * 0.125 // 1)),
            # nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(int(embedding_dim * 0.125 // 1), int(output_dim)),
        )
        # Initialize the Linear layers in relation_network
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    embedding_dim = 128
    output_dim = 4
    n_samples = 4
    n_tuple_network = NTupleNetwork(embedding_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(n_tuple_network.parameters(), lr=0.01)

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
