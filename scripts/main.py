import sklearn.metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as utils

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 100
hidden_size = 100
num_classes = 2
num_epochs = 1000
batch_size = 100
learning_rate = 0.001

data = np.load("output/ppi_as_vec.npy")
tensor_x = data[:, :-1]
tensor_y = data[:, -1:]

# Build train and test sets
N = int(len(tensor_y))
test_size = int(N/10)
data_ixs = np.random.permutation(np.arange(N))
X_test = tensor_x[data_ixs[:test_size], :]
y_test = tensor_y[data_ixs[:test_size], :]
X_train = tensor_x[data_ixs[test_size:], :]
y_train = tensor_y[data_ixs[test_size:], :]

X_train = torch.from_numpy(np.float32(X_train))
y_train = torch.from_numpy(np.int_(y_train.ravel()))
X_test = torch.from_numpy(np.float32(X_test))
y_test = torch.from_numpy(np.int_(y_test.ravel()))

# Transform to pytorch tensor
train_dataset = utils.TensorDataset(X_train, y_train)
test_dataset = utils.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        #self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_and_test():
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            # Move tensors to the configured device
            #images = images.reshape(-1, 28*28).to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model and compute metrics
    with torch.no_grad():
        model_predictions = []
        labels = []

        for X_test, y_test in test_loader:
            outputs = model(X_test)
            labels.extend(y_test)
            model_prediction = list(torch.max(outputs, 1)[1].detach().cpu().numpy())
            model_predictions.extend(model_prediction)

        print("Test Accuracy: " + str(sklearn.metrics.accuracy_score(labels, model_predictions)))
        print("Test Precision: " + str(sklearn.metrics.precision_score(labels, model_predictions)))
        print("Test Recall: " + str(sklearn.metrics.recall_score(labels, model_predictions)))
        print("Test F1 Score: " + str(sklearn.metrics.f1_score(labels, model_predictions)))
        print("Test Confusion Matrix: " + str(sklearn.metrics.confusion_matrix(labels, model_predictions)))
        print("Test AUC: " + str(roc_auc_score(labels, model_predictions)))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')


#train_and_test()

