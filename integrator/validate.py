import torch
from integrator.objects import data_gen, model, criterion
from sklearn.metrics import f1_score

model.eval()
with torch.no_grad():
    valid_data, valid_labels = data_gen.generate_data()  # ideally use separate method for validation data
    valid_outputs = model(valid_data)
    valid_loss = criterion(valid_outputs, valid_labels)
    _, valid_predicted = torch.max(valid_outputs, 1)
    valid_fscore = f1_score(valid_labels.numpy(), valid_predicted.numpy(), average='macro')

print(f"Validation Loss: {valid_loss.item()}, Validation F-Score: {valid_fscore}")