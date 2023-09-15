import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam

from integrator.config import MODEL_PATH, OPTIMIZER_PATH
from integrator.selector import DataGenerator
from integrator.teacher import Classifier

# Initialization
model = Classifier()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
data_gen = DataGenerator()
epochs = 5

# Train the model
for epoch in range(epochs):
    model.train()

    # Fetch training data
    train_data, train_labels = data_gen.generate_data()

    # Forward pass
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate F-score for training data
    _, predicted = torch.max(outputs, 1)
    train_fscore = f1_score(train_labels.numpy(), predicted.numpy(), average="macro")

    # Fetch validation data
    (
        valid_data,
        valid_labels,
    ) = data_gen.generate_data()  # ideally use separate method for validation data
    model.eval()
    with torch.no_grad():
        valid_outputs = model(valid_data)
        valid_loss = criterion(valid_outputs, valid_labels)
        _, valid_predicted = torch.max(valid_outputs, 1)
        valid_fscore = f1_score(
            valid_labels.numpy(), valid_predicted.numpy(), average="macro"
        )

    # Adjust data distribution
    data_gen.adjust_data_distribution(train_fscore)

    print(
        f"Epoch {epoch+1}, Loss: {loss.item()}, F-Score: {train_fscore}, Validation Loss: {valid_loss.item()}, Validation F-Score: {valid_fscore}"
    )

# Save the model and optimizer
torch.save(model.state_dict(), MODEL_PATH)
torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

# Later, to load and validate
model = Classifier()
optimizer = Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load(MODEL_PATH))
optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
