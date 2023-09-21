import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def assign_labels(output_logits):
    batch_size, sample_size, num_labels = output_logits.shape

    all_assigned_labels = []

    for b in range(batch_size):
        probs = torch.softmax(output_logits[b], dim=1)
        max_probs, max_labels = torch.max(probs, dim=1)

        # Sort indices by maximum probability.
        sorted_indices = torch.argsort(max_probs, descending=True)

        assigned = set()
        final_labels = [-1] * sample_size

        for idx in sorted_indices:
            label = max_labels[idx].item()

            if label == 0:
                final_labels[idx] = 0
            else:
                if label not in assigned:
                    final_labels[idx] = label
                    assigned.add(label)
                else:
                    final_labels[idx] = 0

        all_assigned_labels.append(final_labels)

    return torch.tensor(all_assigned_labels)


if __name__ == "__main__":
    # Test with the provided tensor
    output_logits = torch.tensor(
        [
            [
                [0.1286, 0.6, 0.0437, -0.0665],  # 1
                [0.9, -0.0980, 0.1643, 0.0702],  # 0
                [0.8, -0.0980, 0.1643, 0.0702],  # 0
                [0.1923, -0.0892, 0.1502, 0.5],  # 3
                [0.1349, -0.1502, 0.4, -0.0061], # 2
            ]
            ,
            [
                [0.1286, 0.6, 0.0437, -0.0665],  # 1
                [0.9, -0.0980, 0.1643, 0.0702],  # 0
                [-0.8, -0.0980, 0.1643, 0.0702],  # 0
                [0.1923, -0.0892, 0.1502, 0.5],  # 3
                [0.1349, -0.1502, 0.4, -0.0061],  # 2
            ],
            [                [0.1349, -0.1502, 0.4, -0.0061],  # 2
                             [0.1923, -0.0892, 0.1502, 0.5],  # 3
                             [-0.8, -0.0980, 0.1643, 0.0702],  # 0
                             [0.9, -0.0980, 0.1643, 0.0702],  # 0

                             [0.1286, 0.6, 0.0437, -0.0665],  # 1
            ]

        ]
    )

    print(assign_labels(output_logits))
