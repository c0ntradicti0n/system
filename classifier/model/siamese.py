import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.functional import pairwise_distance


def asymmetric_similarity(tensor_a, tensor_b):
    similarity = pairwise_distance(tensor_a - tensor_a * 0.2, tensor_b, p=2)

    return similarity


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm((1, embedding_dim)),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm((1, embedding_dim)),
        )
        # Initialize the Linear layers
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        return self.fc(x)


def triplet_loss(anchor, positive, negative, margin=1):
    distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    distance_negative = torch.norm(anchor - negative, p=2, dim=1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    result = torch.mean(losses)
    if losses.shape[0] == 0 or result.isnan():
        print("Losses shape is 0")
        return losses
    return result


def evaluate_model(model, triplets_embeddings, threshold=10):
    model.eval()
    with torch.no_grad():
        distances = []
        labels = []  # 1 for similar, 0 for dissimilar
        for embeddings in triplets_embeddings:
            anchor, positive, negative = embeddings
            anchor_output = model(anchor.unsqueeze(0))
            positive_output = model(positive.unsqueeze(0))
            negative_output = model(negative.unsqueeze(0))

            pos_dist = pairwise_distance(anchor_output, positive_output, 2).item()
            neg_dist = pairwise_distance(anchor_output, negative_output, 2).item()

            distances.append(pos_dist)
            labels.append(1)  # Similar

            distances.append(neg_dist)
            labels.append(0)  # Dissimilar


        distances = np.array(distances)
        threshold = np.mean(distances)
        predictions = (distances < threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        return precision, recall, f1, threshold
