import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam

from integrator import config
from integrator.config import MODEL_PATH, OPTIMIZER_PATH
from integrator.selector import DataGenerator
from integrator.teacher import Classifier



# Initialization
model = Classifier()
optimizer = Adam(model.parameters(), lr=0.007)
criterion = nn.CrossEntropyLoss()
data_gen = DataGenerator()
epochs = 1000

# Train the model
for epoch in range(epochs):
    # train the model

    model.train()

    # Fetch training data
    train_data, train_labels = data_gen.generate_data()

    # Forward pass
    outputs = model(train_data)

    # Reshape outputs and labels
    outputs_reshaped = outputs.view(-1, 4)  # Change here to flatten both batch and sequence dimensions
    train_labels_reshaped = train_labels.view(-1)  # Flatten the labels


    loss = criterion(outputs_reshaped, train_labels_reshaped)  # Use reshaped tensors

    # Backward pass and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get sigmoid outputs and predictions
    sigmoid_outputs = torch.sigmoid(outputs)
    predicted = (sigmoid_outputs > 0.5).int()

    # ...
    predicted_reshaped = predicted.view(config.batch_size * config.n_samples, 4)  # Flatten both batch and sequence dimensions
    train_labels_reshaped = train_labels.view(-1)  # Flatten the labels
    predicted_labels =torch.argmax(predicted_reshaped, dim=1)

    # Calculate F-score for training data using reshaped tensors
    train_fscore = f1_score(train_labels_reshaped.numpy(), predicted_labels.numpy(), average="macro")


    # Fetch validation data
    valid_data, valid_labels = data_gen.generate_data()

    model.eval()
    with torch.no_grad():
        valid_outputs = model(valid_data)

        # Reshape outputs and labels just like we did for training
        valid_outputs_reshaped = valid_outputs.view(-1, 4)
        valid_labels_reshaped = valid_labels.view(-1)

        valid_loss = criterion(valid_outputs_reshaped, valid_labels_reshaped)

        # Get sigmoid outputs and predictions
        sigmoid_valid_outputs = torch.sigmoid(valid_outputs_reshaped)
        valid_predicted_reshaped = (sigmoid_valid_outputs > 0.5).int()

        # Convert to class labels
        valid_predicted_labels = torch.argmax(valid_predicted_reshaped, dim=1)

        valid_fscore = f1_score(valid_labels_reshaped.numpy(), valid_predicted_labels.numpy(), average="macro")

    # Adjust data distribution
    data_gen.adjust_data_distribution(train_fscore)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item()}, F-Score: {train_fscore}, Validation Loss: {valid_loss.item()}, Validation F-Score: {valid_fscore}"
        )
        print(f"Predicted: {predicted_labels}")

# Save the model and optimizer
torch.save(model.state_dict(), MODEL_PATH)
torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

# Later, to load and validate
model = Classifier()
optimizer = Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load(MODEL_PATH))
optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
