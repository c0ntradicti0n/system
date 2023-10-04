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
            mean=means[i], cov=covs[i], size=num_samples * 4
        )
        samples.extend(sample)
        labels.extend([i] * (num_samples * 4))

    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(
        np.array(labels), dtype=torch.long
    )

class SiameseNTupleNetwork(nn.Module):
    def __init__(self, embedding_dim, output_dim, n_samples):
        super(SiameseNTupleNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_samples = n_samples

        # Shared network for pairwise comparisons
        self.shared_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        # Classifier
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim , embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, x):
        comparisons = []
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                if i != j:
                    diff = x[i] - x[j]
                    comparisons.append(self.shared_net(diff))

        print(len(comparisons))
        # Stack the comparisons to create a 2D tensor
        stacked = torch.stack(comparisons, dim=0)

        # Reshape the 2D tensor to retain the batch dimension
        # Classify
        output = self.classifier(stacked)
        return output


if __name__ == "__main__":
    embedding_dim = 128
    output_dim = 4
    n_samples = 4
    siamese_network = SiameseNTupleNetwork(embedding_dim, output_dim, n_samples)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(siamese_network.parameters(), lr=0.001)

    samples, labels = generate_samples()

    for epoch in range(100):
        optimizer.zero_grad()

        indices = np.random.choice(len(samples), size=n_samples, replace=False)
        selected_samples = torch.stack([samples[index] for index in indices])
        selected_labels = torch.tensor([labels[index] for index in indices], dtype=torch.long)

        outputs = siamese_network(selected_samples).float()
        outputs = outputs.view(n_samples, n_samples - 1, output_dim).mean(dim=1)

        selected_labels.reshape(-1)
        print (outputs.shape, selected_labels.shape)
        print(outputs.shape, outputs.dtype)
        print(selected_labels.shape, selected_labels.dtype)

        loss = criterion(outputs, selected_labels)
        loss.backward()

        optimizer.step()

        # Evaluation
        with torch.no_grad():
            outputs = siamese_network(selected_samples)
            outputs = outputs.view(n_samples, n_samples - 1, output_dim).mean(dim=1)

            _, predicted = torch.max(outputs, 1)
            f1 = f1_score(predicted, selected_labels, average="micro")
            print(f"Epoch {epoch + 1}, F1 Score: {f1:.4f}, Loss: {loss.item()}")


