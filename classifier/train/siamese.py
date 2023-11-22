import importlib
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR

from classifier.model.different_models import get_model_config
from classifier.model.siamese import (SiameseNetwork, evaluate_model,
                                      triplet_loss)
from lib import embedding
from lib.random_take_from_generator import random_from_generator


def train(config_name):

    config = get_model_config(config_name)


    samples = importlib.import_module(config.samples).samples

    random_gen = random_from_generator(samples)

    triplets = (
        (a, b, samples[random.choice(range(len(samples)))][0] if isinstance(samples, list) else next(random_gen)[random.choice([0,1])])
        for i, (a, b) in enumerate(samples)
    )


    model = SiameseNetwork(1024)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.03
    )
    # meandering learning rate, that gets smaller over time
    scheduler = CyclicLR(
        optimizer,
        mode="exp_range",
        gamma=0.98,
        base_lr=0.000000001,
        max_lr=0.03,
        step_size_up=10,
        cycle_momentum=False,
    )
    best_f1 = 0


    for epoch in range(config.n_epochs):
        train_data = [next(triplets) for _ in range(config.batch_size)]
        random.shuffle(train_data)
        triplets_embeddings = embedding.get_embeddings(train_data)

        model.train()

        # Use mini-batches instead of single triplets
        batch_size = config.batch_size
        total_loss = 0
        for _ in range(len(triplets_embeddings) // batch_size):
            optimizer.zero_grad()
            batch_indices = np.random.choice(
                len(triplets_embeddings), size=batch_size, replace=False
            )
            batch_loss = 0

            for idx in batch_indices:
                anchor, positive, negative = triplets_embeddings[idx]

                anchor_output = model(anchor.unsqueeze(0))
                positive_output = model(positive.unsqueeze(0))
                negative_output = model(negative.unsqueeze(0))

                loss = triplet_loss(
                    anchor_output, positive_output, negative_output
                )
                batch_loss += loss / batch_size

            batch_loss.backward()

            optimizer.step()

            total_loss += batch_loss.item()
        scheduler.step()


        precision, recall, f1 = evaluate_model(model, triplets_embeddings)
        avg_loss = total_loss
        print(
            f"{config_name} Epoch {epoch + 1}, Avg Loss: {total_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.2E}"
        )
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    config.MODEL_DIR, f"f1v={f1:.2f}-" + config.MODEL_PATH
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    config.MODEL_DIR, f"f1v={f1:.2f}-" + config.OPTIMIZER_PATH
                ),
            )
