import sklearn.metrics
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import numpy as np
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


def train(model, train_data_loader, val_data_loader, device, num_epochs=50):

    epochs = []
    losses ={'train': [], 'validation': []}
    accuracies = {'train': [], 'validation': []}
    N_train = train_data_loader.dataset.tensors[0].shape[0]
    N_val = val_data_loader.dataset.tensors[0].shape[0]

    # Train and Validate
    for epoch in range(num_epochs):

        print(f"epoch: {epoch + 1}")

        ''' TRAINING '''
        tmp_loss = 0
        predictions = []
        labels = []

        for i, (x, y) in enumerate(train_data_loader):

            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = model.criterion(outputs, y)

            # Backward and optimize
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # update training loss and prediction
            tmp_loss += loss.item()
            predictions.extend(torch.max(outputs, 1)[1].detach().numpy())
            labels.extend(y.detach().numpy())

        # update total loss and accuracy
        losses['train'].append(tmp_loss/N_train)
        acc = sklearn.metrics.accuracy_score(labels, predictions)
        accuracies['train'].append(acc)

        print(f"train loss: {tmp_loss/N_train}\ttrain accuracy: {acc}")

        ''' VALIDATION '''
        tmp_loss = 0
        predictions = []
        labels = []

        for i, (x, y) in enumerate(val_data_loader):

            # predict
            outputs = model(x)
            loss = model.criterion(outputs, y)

            # update validation loss and accuracy
            tmp_loss += loss.item()
            predictions.extend(torch.max(outputs, 1)[1].detach().numpy())
            labels.extend(y.detach().numpy())

        losses['validation'].append(tmp_loss/N_val)
        acc = sklearn.metrics.accuracy_score(labels, predictions)
        accuracies['validation'].append(acc)

        print(f"val loss: {tmp_loss/N_val}\tval accuracy: {acc}")

        epochs.append(epoch + 1)

    return epochs, losses, accuracies


def predict(model, data_loader):
    # Predict model and compute metrics
    with torch.no_grad():
        model_predictions = []
        labels = []

        for X, y in data_loader:
            outputs = model(X)
            labels.extend(y)
            model_prediction = list(torch.max(outputs, 1)[1].detach().numpy())
            model_predictions.extend(model_prediction)

        metrics = {"accuracy": sklearn.metrics.accuracy_score(labels, model_predictions),
                   "precision": sklearn.metrics.precision_score(labels, model_predictions),
                   "recall": sklearn.metrics.recall_score(labels, model_predictions),
                   "F1": sklearn.metrics.f1_score(labels, model_predictions),
                   "AUC": roc_auc_score(labels, model_predictions)}
        confusion = sklearn.metrics.confusion_matrix(labels, model_predictions)

        return metrics, confusion
