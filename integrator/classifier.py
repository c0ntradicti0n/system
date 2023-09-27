import torch
from scipy.stats import multivariate_normal
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


def generate_samples(num_samples=1000, embedding_dim=128):
    means = [np.random.rand(embedding_dim) * i for i in range(4)]
    covs = [np.eye(embedding_dim) * 0.1 for _ in range(4)]
    labels = []
    samples = []

    for i in range(4):
        sample = multivariate_normal.rvs(mean=means[i], cov=covs[i], size=num_samples // 4)
        samples.extend(sample)
        labels.extend([i] * (num_samples // 4))

    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)


class NTupleNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(NTupleNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward_one(self, x):
        out = self.fc(x)
        return out

    def forward(self, samples):
        outputs = torch.stack([self.forward_one(sample) for sample in samples])
        return outputs


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
            f1 = f1_score(labels, predicted, average='micro')
            print(f"Epoch {epoch + 1}, F1 Score: {f1:.4f}, Loss: {loss.item()}")