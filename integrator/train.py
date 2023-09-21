import os

import torch
from pytorch_lamb import Lamb
from pytorch_optimizer import AdaBelief
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from integrator import config
from integrator.classifier import SiameseNetwork
from integrator.config import MODEL_PATH, OPTIMIZER_PATH
from integrator.exclusive_labels import assign_labels
from integrator.selector import DataGenerator


def colorized_comparison(predicted_labels, gold_labels):
    ANSI_RED = "\033[91m"
    ANSI_GREEN = "\033[92m"
    ANSI_RESET = "\033[0m"

    comparison_results = []

    # Ensure the tensors are on the CPU and detached from the computation graph
    predicted_labels = predicted_labels.cpu().detach().numpy()
    gold_labels = gold_labels.cpu().detach().numpy()

    for pred, true in zip(predicted_labels, gold_labels):
        if pred == true:
            comparison_results.append(f"{ANSI_GREEN}{pred}{ANSI_RESET}")
        else:
            comparison_results.append(f"{ANSI_RED}{pred}{ANSI_RESET}")
    return "\n".join(["".join([str(i) for i in gold_labels]),
    ("".join(comparison_results))])


# Initialization
model = SiameseNetwork(config.embedding_dim, config.n_classes)
optimizer = Adam(model.parameters(),  weight_decay= 0.00001)
criterion = nn.CrossEntropyLoss()
data_gen = DataGenerator()
epochs = 10000
scheduler = ReduceLROnPlateau(
    optimizer, "min", patience=10, factor=0.7, verbose=True
)  # Reduce the LR if validation loss doesn't improve for 10 epochs


max_fscore = 0
old_gradient_norm = 0
if not os.path.exists("models"):
    os.makedirs("models")
# Train the model
for epoch in range(epochs):
    model.train()

    # Fetch training data
    train_data, train_labels = data_gen.generate_data()

    # Forward pass
    input_data = train_data.view(config.n_samples, -1, config.embedding_dim)
    optimizer.zero_grad()

    outputs = model(*input_data)

    # Reshape outputs and labels
    outputs_reshaped = outputs.view(-1, config.n_classes)

    train_labels_reshaped = train_labels.view(-1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs_reshaped, train_labels_reshaped)

    # Backward pass and optimizer step
    loss.backward()
    optimizer.step()

    train_labels_reshaped = train_labels.view(-1)  # Flatten the labels
    predicted_labels = torch.argmax(outputs, dim=-1)
    # print(f"{predicted.shape=}, {sigmoid_outputs.shape=}, {predicted_reshaped.shape=}, {train_labels_reshaped.shape=}, {predicted_labels.shape=}")

    # Calculate F-score for training data using reshaped tensors
    train_fscore = f1_score(
        train_labels.view(-1).numpy(),
        predicted_labels.view(-1).numpy(),
        average="macro",
    )

    # Fetch validation data
    valid_data, valid_labels = data_gen.generate_data()

    model.eval()

    with torch.no_grad():
        input_data = valid_data.view(config.n_samples, -1, config.embedding_dim)

        valid_outputs = model(*input_data)

        # Reshape outputs and labels
        outputs_reshaped = valid_outputs.view(-1, config.n_classes)
        valid_labels_reshaped = valid_labels.view(-1)

        predicted_labels = torch.argmax(valid_outputs, dim=-1)

        valid_loss = criterion(outputs_reshaped, valid_labels_reshaped)

        valid_fscore = f1_score(
            valid_labels.view(-1).numpy(),
            predicted_labels.view(-1).numpy(),
            average="macro",
        )

        scheduler.step(valid_loss)

    # Adjust data distribution
    data_gen.adjust_data_distribution(train_fscore)

    if valid_fscore > max_fscore:
        print(
            f"Epoch {epoch + 1}, loss: {loss.item():.2f}, f1-valid: {valid_fscore:.2f}, f1-train: {train_fscore:.2f}, lr: {optimizer.param_groups[0]['lr']}"
        )
        max_fscore = valid_fscore
        torch.save(model.state_dict(), "models/"+ f"f1v={valid_fscore:.2f}-" + MODEL_PATH)
        torch.save(optimizer.state_dict(), "models/"+ f"f1v={valid_fscore:.2f}-" +  OPTIMIZER_PATH)
        with open("models/"+ f"f1v={valid_fscore:.2f}-comp" +  ".txt", "w") as f:
            f.write(f"Epoch {epoch + 1}, loss: {loss.item():.2f}, f1-valid: {valid_fscore:.2f}, f1-train: {train_fscore:.2f}, lr: {optimizer.param_groups[0]['lr']}")
            f.write("\n")
            f.write(colorized_comparison(predicted_labels.view(-1), valid_labels_reshaped))
    print (colorized_comparison(predicted_labels.view(-1), valid_labels_reshaped))
    # print("".join(valid_labels_reshaped.detach().numpy().astype(str)))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)


    print(
        f"f={max_fscore:.2f}"
        f" {epoch=}, {loss=}, {train_fscore=:.2f}, {valid_fscore=:.2f}"
        f" {grad_norm=}"
    )
