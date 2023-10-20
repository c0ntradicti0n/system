import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from classifier.model import NTupleNetwork


def get_model(config):
    return NTupleNetwork(config.embedding_dim, config.n_classes)


def get_prediction(model, input_data, config, compute_confidence=False):
    input_reshaped = input_data.view(-1, config.n_samples, config.embedding_dim)
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

    for logits in outputs:
        # Apply Hungarian assignment to determine class assignment for each sample in the set
        assignment = hungarian_assignment(logits.detach().cpu().numpy())

        # Assign class labels based on the Hungarian assignment
        predicted_labels = torch.tensor(assignment, dtype=torch.long).to(logits.device)

        # Reshape the logits to match the original shape
        logits_reshaped = logits.view(-1, config.n_classes)

        # Append results to the lists
        all_predicted_labels.append(predicted_labels)
        all_outputs_reshaped.append(logits_reshaped)

    # Stack the results back into tensors
    all_predicted_labels = torch.cat(all_predicted_labels)
    all_outputs_reshaped = torch.cat(all_outputs_reshaped)

    # Convert to a PyTorch tensor
    train_predicted_labels = torch.tensor(all_predicted_labels)
    if compute_confidence:
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(all_outputs_reshaped, dim=1)

        # Get the maximum probability for each sample
        confidence_scores = probabilities.max(dim=1).values
        return train_predicted_labels, confidence_scores
    return train_predicted_labels, outputs_reshaped


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
