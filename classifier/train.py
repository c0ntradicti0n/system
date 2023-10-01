import logging
import os

import torch
from different_models import gen_config
from model import NTupleNetwork
from selector import DataGenerator
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR


def colorized_comparison(prefix, predicted_labels, gold_labels):
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
    return "\n".join(
        [
            prefix + "".join([str(i) for i in gold_labels]),
            (prefix + "".join(comparison_results)),
        ]
    )


for config in gen_config():
    model = NTupleNetwork(config.embedding_dim, config.n_classes)
    optimizer = Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    data_gen = DataGenerator(config)
    epochs = config.n_epochs
    scheduler = CyclicLR(
        optimizer,
        base_lr=0.001,
        max_lr=0.006,
        step_size_up=2000,
        mode="triangular",
        cycle_momentum=False,
    )
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.MODEL_DIR, "runs/experiment_1"))

    max_fscore = 0
    old_gradient_norm = 0
    best_lr = 0.003  # Initial learning rate
    counter = 0  # Counter to track the number of epochs since the last best F-score

    if not os.path.exists("models"):
        os.makedirs("models")
    # Train the model
    for epoch in range(epochs):
        model.train()

        # Fetch training data
        train_data, train_labels, texts = data_gen.generate_data()

        # Forward pass
        input_data = train_data.view(config.n_samples, -1, config.embedding_dim)
        optimizer.zero_grad()
        if epoch == 0:
            writer.add_graph(model, input_data)

        outputs = model(input_data)

        # Reshape outputs and labels
        outputs_reshaped = outputs.view(-1, config.n_classes)
        train_labels_reshaped = train_labels.view(-1)

        loss = criterion(outputs_reshaped, train_labels_reshaped)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        train_predicted_labels = torch.argmax(outputs, dim=-1)

        # Calculate F-score for training data using reshaped tensors
        train_fscore = f1_score(
            train_labels_reshaped.numpy(),
            train_predicted_labels.view(-1).numpy(),
            average="macro",
        )

        # Fetch validation data
        valid_data, valid_labels, texts = data_gen.generate_data()

        model.eval()

        with torch.no_grad():
            input_data = valid_data.view(config.n_samples, -1, config.embedding_dim)
            valid_outputs = model(input_data)

            # Reshape outputs and labels
            outputs_reshaped = valid_outputs.view(-1, config.n_classes)
            valid_labels_reshaped = valid_labels.view(-1)

            predicted_labels = torch.argmax(valid_outputs, dim=-1)

            valid_loss = criterion(outputs_reshaped, valid_labels_reshaped)

            valid_fscore = f1_score(
                valid_labels_reshaped.numpy(),
                predicted_labels.view(-1).numpy(),
                average="macro",
            )

            # Convert tensor to numpy for sklearn metrics
            predicted_labels_np = predicted_labels.cpu().numpy().ravel()
            true_labels_np = valid_labels_reshaped.cpu().numpy().ravel()

            # Calculate accuracy, precision, and recall
            accuracy = accuracy_score(true_labels_np, predicted_labels_np)
            precision = precision_score(
                true_labels_np, predicted_labels_np, average="macro"
            )
            recall = recall_score(true_labels_np, predicted_labels_np, average="macro")

        if valid_fscore > max_fscore:
            print(
                f"Epoch {epoch + 1}, loss: {loss.item():.2f}, f1-valid: {valid_fscore:.2f}, f1-train: {train_fscore:.2f}, lr: {optimizer.param_groups[0]['lr']}"
            )
            max_fscore = valid_fscore
            torch.save(
                model.state_dict(),
                os.path.join(
                    config.MODEL_DIR, f"f1v={valid_fscore:.2f}-" + config.MODEL_PATH
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    config.MODEL_DIR, f"f1v={valid_fscore:.2f}-" + config.OPTIMIZER_PATH
                ),
            )
            with open(
                os.path.join(config.MODEL_DIR, f"f1v={valid_fscore:.2f}-" + ".txt"), "w"
            ) as f:
                f.write(
                    f"Epoch {epoch + 1}, loss: {loss.item():.2f}, f1-valid: {valid_fscore:.2f}, f1-train: {train_fscore:.2f}, lr: {optimizer.param_groups[0]['lr']}"
                )
                f.write("\n")
                f.write(
                    colorized_comparison(
                        "v: ", predicted_labels.view(-1), valid_labels_reshaped
                    )
                )
                f.write("\n")
                f.write("\n")
                f.write(str(texts))
            counter = 0  # Reset the counter
        else:
            counter += 1  # Increment the counter

        if counter >= 17:  # If n epochs have passed since the last best F-score
            # Reload the last best model and optimizer state
            model.load_state_dict(
                os.path.join(
                    config.MODEL_DIR, f"f1v={max_fscore:.2f}-" + config.MODEL_PATH
                )
            )
            optimizer.load_state_dict(
                torch.load("models/" + f"f1v={max_fscore:.2f}-" + config.OPTIMIZER_PATH)
            )
            optimizer.param_groups[0]["lr"] = best_lr  # Reset the learning rate
            counter = 0  # Reset the counter

        scheduler.step(valid_loss)

        print(
            colorized_comparison(
                "t: ", predicted_labels.view(-1), valid_labels_reshaped
            )
        )
        print(
            colorized_comparison(
                "v: ", train_predicted_labels.view(-1), train_labels_reshaped
            )
        )

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Log parameters to TensorBoard
        writer.add_scalar("Loss", loss.item(), epoch)
        writer.add_scalar("Train F-score", train_fscore, epoch)
        writer.add_scalar("Validation F-score", valid_fscore, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Gradient Norm", grad_norm.item(), epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("Precision", precision, epoch)
        writer.add_scalar("Recall", recall, epoch)

        # Assuming `predictions` are the model predictions and `labels` are the true labels
        writer.add_pr_curve(
            "precision_recall_valid",
            predicted_labels.view(-1),
            valid_labels_reshaped,
            epoch,
        )
        writer.add_pr_curve(
            "precision_recall_valid",
            train_predicted_labels.view(-1),
            train_labels_reshaped,
            epoch,
        )

        hparams = {"lr": 0.1, "batch_size": 64}
        metrics = {"accuracy": accuracy, "loss": loss}
        writer.add_hparams(hparams, metrics)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        print(
            f"f={max_fscore:.2f}"
            f" {epoch=}, {loss=}, {train_fscore=:.2f}, {valid_fscore=:.2f}"
            f" {grad_norm=}"
        )

        if max_fscore > config.stop_f1_score:
            logging.info(
                f"F-score > {config.stop_f1_score=}\n   stopping training {config=}"
            )
            break
