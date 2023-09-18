import torch
from torch import nn
from torch.nn import init

from integrator import config


class MultiInputNetwork(nn.Module):
    def __init__(self):
        super(MultiInputNetwork, self).__init__()
        self.embedding_size = config.embedding_dim

        # Number of relations for n embeddings
        self.num_relations = config.n_samples * (config.n_samples - 1) // 2

        # Layers to process each relation vector
        self.relation_network = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        init.kaiming_normal_(self.relation_network[0].weight)
        init.kaiming_normal_(self.relation_network[3].weight)
        init.kaiming_normal_(self.relation_network[5].weight)


        # Classifier after pooling the relations
        self.classifier = nn.Sequential(
            nn.Linear(
                self.embedding_size, 256
            ),  # This assumes we're reducing each relation to a size of 128 and then pooling
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(
                256, 128
            ),  # This assumes we're reducing each relation to a size of 128 and then pooling
            nn.ReLU(),
            nn.Linear(
                128, 32
            ),  # This assumes we're reducing each relation to a size of 128 and then pooling
            nn.ReLU(),
            nn.Linear(32, config.n_classes),
        )
        init.kaiming_normal_(self.classifier[0].weight)
        init.kaiming_normal_(self.classifier[3].weight)
        init.kaiming_normal_(self.classifier[5].weight)
        init.kaiming_normal_(self.classifier[7].weight)

    def forward(self, *embeddings):
        outputs = []

        # Loop over each embedding to make predictions individually
        for embed in embeddings:
            # Compute pairwise relation vectors using subtraction for this particular embedding
            relations = [
                other - embed for other in embeddings if not torch.equal(embed, other)
            ]
            relations = torch.stack(relations, dim=1)  # Stack along a new dimension

            # Pass each relation through the relation_network
            processed_relations = [
                self.relation_network(rel) for rel in relations.split(1, dim=1)
            ]
            processed_relations = torch.cat(
                processed_relations, dim=1
            )  # Concatenate along feature dimension

            # Pooling over processed relations
            pooled_relations, _ = torch.max(processed_relations, dim=1)


            # Get final output
            output = self.classifier(pooled_relations)
            outputs.append(output)

        # Stack outputs along the new dimension
        outputs = torch.stack(outputs, dim=1)
        return outputs
