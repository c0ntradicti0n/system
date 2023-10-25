import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from classifier.model import NTupleNetwork


def get_model(config):
    return NTupleNetwork(config.embedding_dim, config.n_classes)


def get_prediction(model, input_data, config, compute_confidence=False, n_samples=None):
    input_reshaped = input_data.view(
        -1, n_samples if n_samples else config.n_samples, config.embedding_dim
    )
    outputs = model(input_reshaped)

    # Reshape outputs and labels
    try:
        outputs_reshaped = outputs.view(-1, config.n_classes)
    except:
        print(f"{outputs=} {outputs.shape=} {config.n_classes=}")
        raise
    # Initialize arrays to store predictions and outputs
    all_predicted_labels = []
    all_outputs_reshaped = []

    for i, logits in enumerate(outputs):
        # Apply Hungarian assignment to determine class assignment for each sample in the set
        assignment = hungarian_assignment(logits.detach().cpu().numpy())

        # Assign class labels based on the Hungarian assignment
        predicted_labels = torch.tensor(assignment, dtype=torch.long).to(logits.device)

        if 0 in predicted_labels:
            predicted_labels.fill_(0)

        # Reshape the logits to match the original shape
        try:
            logits_reshaped = logits.view(-1, config.n_samples, config.n_classes)
        except:
            print(
                f"{i=}\n"
                f"{outputs.shape=}\n"
                f"{input_data.shape=}\n"
                f"{logits.shape=}\n"
                f"{config.n_classes=}\n"
                f"{predicted_labels=}\n"
                f"{predicted_labels.shape=}\n"
                f"{assignment=}"
            )
            logits_reshaped = logits.view(-1, config.n_samples, config.n_classes -1)
        all_predicted_labels.append(predicted_labels)
        all_outputs_reshaped.append(logits_reshaped)

    all_predicted_labels = torch.cat(all_predicted_labels)
    all_outputs_reshaped = torch.cat(all_outputs_reshaped)

    if compute_confidence:
        # Apply softmax to convert logits to probabilities
        outputs_without_0 = all_outputs_reshaped[:, :, 1:]
        probabilities = F.softmax(outputs_without_0, dim=1)

        # Get the maximum probability for each sample
        confidence_scores = probabilities.max(dim=1).values
        return all_predicted_labels, confidence_scores

    return all_predicted_labels, outputs_reshaped


def hungarian_assignment(scores):
    # Convert scores to a cost matrix by taking the negative of the scores
    cost_matrix = -scores

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return col_ind


if __name__ == "__main__":
    # Example usage
    import torch

    # Sample scores from your model (replace this with your actual scores)
    scores = torch.tensor([[1, 4, 6, 6, 1], [2, 3, 5, 6, 2]])

    # Apply the Hungarian assignment
    assignment = hungarian_assignment(scores.numpy())

    # The 'assignment' array contains the assigned class for each sample
    print(assignment)
