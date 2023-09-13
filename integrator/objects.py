import torch
from torch.optim import Adam
from sklearn.metrics import f1_score

from integrator.selector import DataGenerator
from integrator.teacher import Classifier
from torch import nn

# Initialization
model = Classifier()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
data_gen = DataGenerator()
epochs = 5