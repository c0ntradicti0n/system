import torch
import torch.nn.functional as F

from classifier.model import NTupleNetwork


def get_model(config):
    return NTupleNetwork(config.embedding_dim, config.n_classes)


def get_prediction(model, input_data, config, compute_confidence=False):
    input_reshaped = input_data.view(config.n_samples, -1, config.embedding_dim)
    outputs = model(input_reshaped)

    # Reshape outputs and labels
    outputs_reshaped = outputs.view(-1, config.n_classes)
    train_predicted_labels = torch.argmax(outputs, dim=-1)
    if compute_confidence:
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(outputs_reshaped, dim=1)

        # Get the maximum probability for each sample
        confidence_scores = probabilities.max(dim=1).values
        return train_predicted_labels, confidence_scores
    return train_predicted_labels, outputs_reshaped
