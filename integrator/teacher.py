import torch
from torch import nn
from integrator import config

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(config.embedding_dim *config.n_samples, 1024)  # 10 samples * 768 embedding size
        self.fc2 = nn.Linear(1024, 4 *config.n_samples )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
