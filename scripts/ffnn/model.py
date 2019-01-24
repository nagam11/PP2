import sklearn.metrics
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.utils.data as utils
import numpy as np


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size=None, num_classes=2, learning_rate=0.001):
        super(NeuralNet, self).__init__()

        if hidden_size == None:
            hidden_size = input_size
        else:
            hidden_size = min(hidden_size, input_size)

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


def train(model, device, train_data_loader, val_data_loader=None, num_epochs=50, debug=False):
    epochs = []
    losses = {'train': [], 'validation': []}
    accuracies = {'train': [], 'validation': []}
    n_train = train_data_loader.dataset.tensors[0].shape[0]
    if val_data_loader is not None:
        n_val = val_data_loader.dataset.tensors[0].shape[0]

    # Train and Validate
    for epoch in range(num_epochs):

        if debug:
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
        losses['train'].append(tmp_loss / n_train)
        acc = sklearn.metrics.accuracy_score(labels, predictions)
        accuracies['train'].append(acc)

        if debug:
            print(f" train loss: {tmp_loss / n_train}\ttrain accuracy: {acc}")

        ''' VALIDATION '''
        # skip validation if no set is given
        if val_data_loader is None:
            continue

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

        losses['validation'].append(tmp_loss / n_val)
        acc = sklearn.metrics.accuracy_score(labels, predictions)
        accuracies['validation'].append(acc)

        if debug:
            print(f" val loss: {tmp_loss / n_val}\tval accuracy: {acc}")

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
                   "AUC": roc_auc_score(labels, model_predictions),
                   'confusion': sklearn.metrics.confusion_matrix(labels, model_predictions)}

        return metrics


def load_data(ppi_protvec):
    """
    load a numpy array containing Protvec vectors and labels
    :param ppi_protvec: filename to numpy array with vectors, last column containing labels
    :return: np arrays for data and labels
    """

    data = np.load(ppi_protvec)
    return data[:, :-1], data[:, -1:]


def create_data_loader(X, y, batch_size=100):
    """
    :param X: data
    :param y: labels
    :return: pytorch datasets for training and testing
    """

    train_dataset = utils.TensorDataset(torch.from_numpy(np.float32(X)),
                                        torch.from_numpy(np.int_(y.ravel())))
    '''# Sampler for DataLoader
    #class_sample_counts = [65365, 24176]
    class_sample_counts = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[y_train]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
                                    weights=samples_weights,
                                    num_samples=len(samples_weights),
                                    replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)                                
    '''
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader
