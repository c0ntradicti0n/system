import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity


def compute_cosine_similarity(embeddings, labels, config):
    batch_size, n_samples = labels.shape
    confidence_scores = torch.zeros(batch_size, n_samples, dtype=torch.float32)

    for batch_idx in range(batch_size):
        for sample_idx in range(n_samples):
            label = labels[batch_idx, sample_idx].item()
            if config == "HIE" and label in [1]:
                # Find the pair for subsumed and subsuming concepts
                pair_label = 3 - label  # 1 becomes 2, 2 becomes 1
                pair_indices = (labels[batch_idx] == pair_label).nonzero(as_tuple=True)[
                    0
                ]
                if pair_indices.nelement() > 0:
                    pair_index = pair_indices[0]
                    confidence_scores[batch_idx, sample_idx] = cosine_similarity(
                        embeddings[batch_idx, sample_idx].unsqueeze(0),
                        embeddings[batch_idx, pair_index].unsqueeze(0),
                    )
            elif config == "TAS" and label in [1, 2, 3]:
                # Compute similarity for thesis-antithesis and synthesis
                thesis_indices = (labels[batch_idx] == 1).nonzero(as_tuple=True)[0]
                antithesis_indices = (labels[batch_idx] == 2).nonzero(as_tuple=True)[0]
                synthesis_indices = (labels[batch_idx] == 3).nonzero(as_tuple=True)[0]
                thesis_embedding = embeddings[batch_idx, thesis_indices].mean(dim=0)

                if (
                    not thesis_indices.nelement() > 0
                    and antithesis_indices.nelement() > 0
                    and synthesis_indices.nelement() > 0
                ):
                    confidence_scores[batch_idx, sample_idx] = -1000
                    continue

                if label == 3:  # Synthesis
                    synthesis_embedding = embeddings[
                        batch_idx, antithesis_indices
                    ].mean(dim=0)
                    mean_embedding = (
                        embeddings[batch_idx, thesis_indices]
                        + embeddings[batch_idx, antithesis_indices]
                    ) / 2
                    cosine_sim = cosine_similarity(
                        embeddings[batch_idx, sample_idx].unsqueeze(0),
                        mean_embedding.mean(dim=0, keepdim=True),
                    )

                    diff_vector_norm = torch.norm(
                        thesis_embedding - synthesis_embedding
                    )

                elif label == 2:  # Antithesis
                    antithesis_embedding = embeddings[
                        batch_idx, antithesis_indices
                    ].mean(dim=0)

                    cosine_sim = cosine_similarity(
                        thesis_embedding.unsqueeze(0),
                        antithesis_embedding.unsqueeze(0),
                    )
                    diff_vector_norm = torch.norm(
                        thesis_embedding - antithesis_embedding
                    )
                    diff_vector_norm = 0
                    cosine_sim = 0

                elif label == 1:
                    cosine_sim = 0
                    diff_vector_norm = 0
                elif label == 0:
                    cosine_sim = 0
                    diff_vector_norm = 0

                confidence_scores[batch_idx, sample_idx] = (
                    cosine_sim + diff_vector_norm / 100
                )

    return confidence_scores


def get_model(config):

    if config.model == "siamese":
        from classifier.model.siamese import SiameseNetwork

        return SiameseNetwork(config.embedding_dim)
    elif config.model == "ntuple":
        from classifier.model.ntuple import NTupleNetwork

        return NTupleNetwork(config.embedding_dim, config.n_classes)
    elif config.model == "som":
        from classifier.model.som import Som

        return Som(config.embedding_dim, config.hidden_dim, config.n_classes)
    else:
        raise ValueError(f"Unknown model {config.model}")


def get_prediction(model, input_data, config, compute_confidence=False, n_samples=None):
    if config.model == "som":
        outputs = model(input_data)

        # For classification tasks, the outputs are usually logits or softmax probabilities
        # Assuming outputs are logits, apply softmax to convert to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get the predicted classes with the highest probability
        predicted_labels = torch.argmax(probabilities, dim=1)

        if config.get("compute_confidence", False):
            confidence_scores = torch.max(probabilities, dim=1)[0] if config.get('compute_confidence', False) else None

            return predicted_labels, confidence_scores
        else:
            return predicted_labels, probabilities

    try:
        input_reshaped = input_data.view(
            -1, n_samples if n_samples else config.n_samples, config.embedding_dim
        )
    except:
        raise
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

    if not config.just_labels:
        # use hungarian assignment to determine class assignment for each sample in the set
        for i, logits in enumerate(outputs):
            # Apply Hungarian assignment to determine class assignment for each sample in the set
            assignment = hungarian_assignment(logits.detach().cpu().numpy())

            # Assign class labels based on the Hungarian assignment
            predicted_labels = torch.tensor(assignment, dtype=torch.long).to(logits.device)

            if 0 in predicted_labels:
                predicted_labels.fill_(0)
            if config.symmetric:
                # sort labels to ensure that the first label is always the positive class
                predicted_labels = torch.sort(predicted_labels)[0]

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
                logits_reshaped = logits.view(-1, config.n_samples, config.n_classes - 1)
            all_predicted_labels.append(predicted_labels)
            all_outputs_reshaped.append(logits_reshaped)
    else:
        # use use argmax to determine class assignment for each sample in the set, classes can be non-exclusive
        predicted_labels = torch.argmax(outputs_reshaped, dim=1)
        all_predicted_labels.append(predicted_labels)

    all_predicted_labels = torch.cat(all_predicted_labels)

    # Replace the existing confidence scoring method with the above function
    if compute_confidence:
        labels_reshaped = all_predicted_labels.view(-1, config.n_samples)
        confidence_scores = compute_cosine_similarity(
            input_reshaped, labels_reshaped, config.name
        )
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
