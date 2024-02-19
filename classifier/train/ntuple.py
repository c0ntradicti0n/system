import logging
import os

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import CyclicLR

from classifier.model.different_models import (get_model_config,
                                               print_model_config)
from classifier.model.siamese import evaluate_model
from classifier.read.selector import DataGenerator
from classifier.result.think import get_model, get_prediction

"""
Use WordNet for more samples

Build networkX graph from Text by linking paragraphs by levels of numbers and sequence of text

Fan Modal for user input

Use Proportional Integral Derivative (PID) controller for adjusting the data connection
"""


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

def train(config_name):

    config = get_model_config(config_name)
    model = get_model(config)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    data_gen = DataGenerator(config)


    print_model_config(config)

    # meandering learning rate, that gets smaller over time
    scheduler = CyclicLR(
        optimizer,
        mode="exp_range",
        gamma=0.99,
        base_lr=0,
        max_lr=0.006,
        step_size_up=config.batches_per_epoch * 0.7,
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
    for epoch in range(config.n_epochs):
        model.train()
        total_loss = 0
        total_train_fscore = 0

        for batch in range(config.batches_per_epoch):
            # Fetch training data
            train_data, train_labels, texts = data_gen.generate_data(config=config)

            # Forward pass
            optimizer.zero_grad()

            if config.model in ["ntuple"]:
                train_predicted_labels, outputs_reshaped = get_prediction(
                    model, train_data, config
                )

                if config.symmetric:
                    train_labels = torch.sort(train_labels)[0]

                train_labels_reshaped = train_labels.view(-1)

                loss = criterion(outputs_reshaped, train_labels_reshaped)
            elif config.model in ["som"]:
                train_predicted_labels, outputs_reshaped = get_prediction(
                    model, train_data, config
                )


                train_labels_reshaped = train_labels[:,1]

                loss = criterion(outputs_reshaped, train_labels_reshaped)
            else:
                loss = 0

                for anchor, positive, negative in train_data:
                    anchor_output = model(anchor.unsqueeze(0))
                    positive_output = model(positive.unsqueeze(0))
                    negative_output = model(negative.unsqueeze(0))

                    sample_loss = triplet_loss(
                        anchor_output, positive_output, negative_output,
                    )
                    loss += sample_loss



            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Calculate F-score for training data using reshaped tensors
            if config.model in ["ntuple"]:
                train_fscore = f1_score(
                    train_labels_reshaped.numpy(),
                    train_predicted_labels.view(-1).numpy(),
                    average=config.get("f1", "macro"),
                )
            elif config.model in ["som"]:
                train_fscore = f1_score(
                    train_labels_reshaped.numpy(),
                    train_predicted_labels.view(-1).numpy(),
                    average=config.get("f1", "macro"),
                )

            else:
                precision, recall, train_fscore = evaluate_model(model, train_data)

            total_loss += loss.item()
            total_train_fscore += train_fscore

            scheduler.step()

            print(
                f"Epoch {epoch + 1}, {batch=}, {loss=}, {train_fscore=:.2f} {optimizer.param_groups[0]['lr']:.2E}"
            )
            if train_fscore > max_fscore and train_fscore > 0.5:
                break

        avg_loss = total_loss / config.batches_per_epoch
        avg_train_fscore = total_train_fscore / config.batches_per_epoch

        print(
            f"Epoch {epoch + 1}, avg loss: {avg_loss:.2f}, avg f1-train: {avg_train_fscore:.2f}"
        )

        # Fetch validation data
        valid_data, valid_labels, texts = data_gen.generate_data(
            config=config, batch_size=config.batch_size * 5
        )
        if config.symmetric:
            valid_labels = torch.sort(valid_labels)[0]

        model.eval()

        if config.model in ["ntuple"]:
            with torch.no_grad():
                predicted_labels, outputs_reshaped = get_prediction(
                    model, valid_data, config
                )

                valid_labels_reshaped = valid_labels.view(-1)

                valid_loss = criterion(outputs_reshaped, valid_labels_reshaped)

                valid_fscore = f1_score(
                    valid_labels_reshaped.numpy(),
                    predicted_labels.view(-1).numpy(),
                    average=config.get("f1", "macro"),
                )

                # Convert tensor to numpy for sklearn metrics
                predicted_labels_np = predicted_labels.cpu().numpy().ravel()
                true_labels_np = valid_labels_reshaped.cpu().numpy().ravel()

                # Calculate accuracy, precision, and recall
                accuracy = accuracy_score(true_labels_np, predicted_labels_np)
                precision = precision_score(
                    true_labels_np, predicted_labels_np,  average=config.get("f1", "macro"),

                )
                recall = recall_score(true_labels_np, predicted_labels_np, average=config.get("f1", "macro"),
)

        if config.model in ["som"]:
            with torch.no_grad():
                predicted_labels, outputs_reshaped = get_prediction(
                    model, valid_data, config
                )

                valid_labels_reshaped = valid_labels[:,1]

                valid_loss = criterion(outputs_reshaped, valid_labels_reshaped)

                valid_fscore = f1_score(
                    valid_labels_reshaped.numpy(),
                    predicted_labels.view(-1).numpy(),
                    average=config.get("f1", "macro"),
                )

                # Convert tensor to numpy for sklearn metrics
                predicted_labels_np = predicted_labels.cpu().numpy().ravel()
                true_labels_np = valid_labels_reshaped.cpu().numpy().ravel()

                # Calculate accuracy, precision, and recall
                accuracy = accuracy_score(true_labels_np, predicted_labels_np)
                precision = precision_score(
                    true_labels_np, predicted_labels_np, average=config.get("f1", "macro"),

                )
                recall = recall_score(true_labels_np, predicted_labels_np, average=config.get("f1", "macro"),
                                      )
        else:
            valid_fscore = train_fscore

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
                if config.model in ["ntuple"]:
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


        if config.model in ["ntuple"] or config.model in ["som"]:
            print(
                colorized_comparison(
                    "v: ", predicted_labels.view(-1), valid_labels_reshaped
                )
            )
            print(
                colorized_comparison(
                    "t: ", train_predicted_labels.view(-1), train_labels_reshaped
                )
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Log parameters to TensorBoard
        writer.add_scalar("Loss", avg_loss, epoch)
        writer.add_scalar("Train F-score", avg_train_fscore, epoch)
        writer.add_scalar("Validation F-score", valid_fscore, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Gradient Norm", grad_norm.item(), epoch)
        if config.model in ["ntuple"]:
            writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("Precision", precision, epoch)
        writer.add_scalar("Recall", recall, epoch)

        if config.model in ["ntuple"] or config.model in ["som"]:
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

        hparams = {
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
        }
        if config.model in ["ntuple"]:
            metrics = {"accuracy": accuracy, "loss": avg_loss}
            writer.add_hparams(hparams, metrics)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        print(
            f"{config_name} f={max_fscore:.2f}"
            f" {epoch=}, {loss=}, {train_fscore=:.2f}, {valid_fscore=:.2f}"
            f" {grad_norm=}"
        )

        if max_fscore > config.stop_f1_score:
            logging.info(
                f"F-score > {config.stop_f1_score=}\n   stopping training {config=}"
            )
            break
