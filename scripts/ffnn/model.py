import sklearn.metrics
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import pprint


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, learning_rate=0.001):
        super(NeuralNet, self).__init__()

        # Model
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)  # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)  # self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # Optimisation
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def train(model, data_loader, device, num_epochs=50, learning_rate=0.001):

    epochs = []
    losses = []
    accuracies = []

    # Train the model
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        tmp_losses = 0
        for i, (x, y) in enumerate(data_loader):
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = model.criterion(outputs, y)

            # Backward and optimize
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            tmp_losses += loss.item()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        epochs.append(epoch + 1)
        print(tmp_losses)
        print(loss.item())
        losses.append(tmp_losses)

    return epochs, losses


def predict(model, data_loader):
    # Predict model and compute metrics
    with torch.no_grad():
        model_predictions = []
        labels = []

        for X, y in data_loader:
            outputs = model(X)
            labels.extend(y)
            model_prediction = list(torch.max(outputs, 1)[1].detach().cpu().numpy())
            model_predictions.extend(model_prediction)

        metrics = {"accuracy": sklearn.metrics.accuracy_score(labels, model_predictions),
                   "precision": sklearn.metrics.precision_score(labels, model_predictions),
                   "recall": sklearn.metrics.recall_score(labels, model_predictions),
                   "F1": sklearn.metrics.f1_score(labels, model_predictions),
                   "AUC": roc_auc_score(labels, model_predictions)}
        pprint.pprint(metrics)
        confusion = sklearn.metrics.confusion_matrix(labels, model_predictions)

        report = str(sklearn.metrics.classification_report(labels, model_predictions,
                                                           labels=[0, 1], target_names=["-", "+"]))
        print(report)

        return metrics, confusion
