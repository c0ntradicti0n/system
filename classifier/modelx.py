import numpy as np
import torch
import torch.nn.functional as F
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
    def __init__(self, embedding_dim, output_dim, n_samples):
        super(NTupleNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.n = n_samples
        self.output_dim = output_dim
        self.n_samples = n_samples

        # Compute the size of pairwise interactions
        self.interaction_dim = n_samples * n_samples

        # Pathway for embeddings
        self.fc_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * n_samples, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
        )

        # Pathway for interactions
        self.fc_interactions = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Final layers
        self.fc_final = nn.Sequential(
            nn.Linear(embedding_dim, (n_samples + 1)),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear((n_samples + 1), output_dim),
        )

        # Initialize weights
        for network in [self.fc_embeddings, self.fc_interactions, self.fc_final]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)

    def compute_interactions(self, x):
        # Compute cosine similarity for pairwise interactions
        interactions = []
        for i in range(self.n):
            for j in range(self.n):
                similarity = F.cosine_similarity(
                    x[i : i + 1, :], x[j : j + 1, :], dim=1, eps=1e-8
                )
                interactions.append(similarity)

        interactions = torch.stack(interactions, dim=1)

        # Pad interactions to match embedding_dim
        padding_size = self.embedding_dim - interactions.size(1)
        if padding_size > 0:
            padding = torch.zeros((interactions.size(0), padding_size), device=x.device)
            interactions = torch.cat([interactions, padding], dim=1)

        return interactions

    def forward(self, x):
        outputs = []
        for samples in x:
            interactions = self.compute_interactions(samples)
            combined = torch.cat([samples, interactions], dim=0)
            out = self.fc_final(combined)
            outputs.append(out)
        return torch.stack(outputs, dim=0)[:, 1:]


if __name__ == "__main__":
    embedding_dim = 128
    output_dim = 4
    n_samples = 4
    batch_size = 16
    n_tuple_network = NTupleNetwork(embedding_dim, output_dim, n_samples=n_samples)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(n_tuple_network.parameters(), lr=0.01)

    samples, labels = generate_samples()

    for epoch in range(1000):
        optimizer.zero_grad()

        selected_samples_list = []
        selected_labels_list = []

        for _ in range(batch_size):
            indices = np.random.choice(len(samples), size=n_samples, replace=False)
            selected_samples = torch.stack([samples[index] for index in indices])
            selected_labels = torch.tensor([labels[index] for index in indices])

            selected_samples_list.append(selected_samples)
            selected_labels_list.append(selected_labels)

        selected_samples = torch.stack(selected_samples_list).view(
            batch_size, n_samples, embedding_dim
        )
        selected_labels = torch.stack(selected_labels_list).view(batch_size, n_samples)

        outputs = n_tuple_network(selected_samples)
        selected_labels = selected_labels.view(batch_size, -1)

        losses = []
        for b in range(batch_size):
            loss = criterion(outputs[b], selected_labels[b])
            losses.append(loss)

        loss = sum(losses) / len(losses)
        loss.backward()

        optimizer.step()

        # Evaluation
        with torch.no_grad():
            outputs = n_tuple_network(selected_samples)
            total_f1 = 0
            for b in range(batch_size):
                _, predicted = torch.max(outputs[b], 1)
                f1 = f1_score(selected_labels[b], predicted, average="micro")
                total_f1 += f1

            avg_f1 = total_f1 / batch_size
            print(
                f"Epoch {epoch + 1}, Average F1 Score: {avg_f1:.4f}, Loss: {loss.item()}"
            )
