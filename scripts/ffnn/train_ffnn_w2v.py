import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import train_test_split

import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-p', '--training_set', required=True)
parser.add_argument('-m', '--model', required=True)  # file path of created model
parser.add_argument('-e', '--num_epochs', required=True, type=int)
parser.add_argument('-b', '--batch_size', required=True, type=int)
args = parser.parse_args()


'''DATA'''
X, y = ffnn.load_data(args.training_set)

'''
# randomise and split
N = int(len(y))
test_size = int(N / 10)
data_ixs = np.random.permutation(np.arange(N))
# train set
train_ix = data_ixs[test_size:]
train_loader = ffnn.create_data_loader(X[train_ix, :], y[train_ix, :])
# test set
test_ix = data_ixs[:test_size]
test_loader = ffnn.create_data_loader(X[test_ix, :], y[test_ix, :])
'''
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)
train_loader = ffnn.create_data_loader(X_train, y_train, batch_size=args.batch_size)
val_loader = ffnn.create_data_loader(X_val, y_val, batch_size=args.batch_size)

'''NEURAL NETWORK'''
dimensions = X.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ffnn.NeuralNet(input_size=dimensions).to(device)
# train model
epochs, losses, accuracies = ffnn.train(model, device, train_loader, val_loader,
                                        num_epochs=args.num_epochs,
                                        debug=True)

# plot losses
training, = plt.plot(epochs, losses['train'], label='training')
validation, = plt.plot(epochs, losses['validation'], label='validation')
plt.legend(handles=[training, validation])
plt.title("Loss over epochs for FFNN")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# plot accuracy
training, = plt.plot(epochs, accuracies['train'], label='training')
validation, = plt.plot(epochs, accuracies['validation'], label='validation')
plt.legend(handles=[training, validation])
plt.title("Accuracy over epochs for FFNN on Protvec")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

metrics = ffnn.predict(model, train_loader)
print('training performance:', metrics)

metrics = ffnn.predict(model, val_loader)
print('training performance:', metrics)

# Save the model checkpoint
torch.save(model.state_dict(), args.model)
