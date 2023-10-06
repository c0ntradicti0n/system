import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau


def generate_samples(num_samples=5000, embedding_dim=128, n_classes = 5):
    means = [np.random.rand(embedding_dim) * i for i in range(4)]
    covs = [np.eye(embedding_dim) * 0.1 for _ in range(4)]
    labels = []
    samples = []

    for i in range(4):
        sample = multivariate_normal.rvs(
            mean=means[i], cov=covs[i], size=num_samples * n_classes
        )
        samples.extend(sample)
        labels.extend([i] * (num_samples * n_classes))

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
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
        )
        # Initialize the Linear layers in relation_network
        for layer in self.shared_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim // 4, embedding_dim // 8),
            nn.ReLU(),
            nn.Linear(embedding_dim // 8, output_dim),
        )
        # Initialize the Linear layers in relation_network
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        batch_size, _, _ = x.shape
        comparisons = []
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                if i != j:
                    diff = x[:, i, :] - x[:, j, :]
                    comparisons.append(self.shared_net(diff))

        stacked = torch.stack(comparisons, dim=1)
        output = self.classifier(stacked.view(batch_size, -1, self.embedding_dim // 4))
        return output


if __name__ == "__main__":
    embedding_dim = 127
    output_dim = 5
    n_samples = 4
    siamese_network = SiameseNTupleNetwork(embedding_dim, output_dim, n_samples)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(siamese_network.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, verbose=True)


    samples, labels = generate_samples(embedding_dim=embedding_dim)

    batch_size = 4200
    for epoch in range(3000):

        optimizer.zero_grad()

        # Randomly select batches
        indices = np.random.choice(len(samples), size=(batch_size, n_samples), replace=False)
        selected_samples = torch.stack([samples[i] for i in indices])
        selected_labels = torch.stack([labels[i] for i in indices])

        outputs = siamese_network(selected_samples).float()
        outputs = outputs.view(batch_size, n_samples, n_samples - 1, output_dim).max(dim=2).values
        outputs = outputs.mean(dim=1)

        loss = criterion(outputs, selected_labels[:, 0])
        loss.backward()

        optimizer.step()
        scheduler.step(loss)


        # Evaluation
        with torch.no_grad():
            outputs = siamese_network(selected_samples)
            outputs = outputs.view(batch_size, n_samples, n_samples - 1, output_dim).max(dim=2).values
            outputs = outputs.mean(dim=1)  # Average over the n_samples dimension

            _, predicted = torch.max(outputs, 1)
            f1 = f1_score(predicted.view(-1), selected_labels[:, 0].view(-1), average="micro")
            print(f"Epoch {epoch + 1}, F1 Score: {f1:.4f}, Loss: {loss.item()} {optimizer.param_groups[0]['lr']}")
