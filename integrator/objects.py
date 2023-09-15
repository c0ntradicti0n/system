from torch import nn
from torch.optim import Adam

from integrator.selector import DataGenerator
from integrator.teacher import Classifier

model = Classifier()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
data_gen = DataGenerator()
epochs = 5
