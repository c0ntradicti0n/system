import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def assign_labels(output_logits):
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(output_logits, dim=2)

    # Remove the probabilities corresponding to label 0
    probs_without_zero = probabilities[:, :, 1:]

    # Convert probabilities to costs
    cost_matrix = 1 - probs_without_zero

    batch_size = cost_matrix.shape[0]
    sample_size = cost_matrix.shape[1]

    all_assigned_labels = []

    for i in range(batch_size):
        # Apply the Hungarian algorithm for each batch
        _, assigned_labels_without_zero = linear_sum_assignment(
            cost_matrix[i].detach().numpy()
        )
        assigned_labels = []

        # Adjust the assigned labels to account for the removed zero label
        for label in assigned_labels_without_zero:
            assigned_labels.append(label + 1)

        # For any remaining unassigned embeddings, assign them label 0
        while len(assigned_labels) < sample_size:
            assigned_labels.append(0)

        all_assigned_labels.append(assigned_labels)

    return torch.tensor(all_assigned_labels)


if __name__ == "__main__":
    # Test with the provided tensor
    output_logits = torch.tensor(
        [
            [
                [0.1286, -0.1598, 0.0437, -0.0665],  # 1
                [0.2352, -0.0980, 0.1643, 0.0702],  # 0
                [0.2352, -0.0980, 0.1643, 0.0702],  # 0
                [0.1923, -0.0892, 0.1502, 0.0461],  # 3
                [0.1349, -0.1502, 0.1297, -0.0061],
            ]  # 2
        ]
    )

    print(assign_labels(output_logits))
