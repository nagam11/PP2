import argparse
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import torch
import numpy as np

import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-p', '--ppi_protvecs', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-e', 'num_epochs', required=True)
args = parser.parse_args()


'''DATA'''
X, y = ffnn.load_data(args.ppi_protvecs)

# randomise and split
N = int(len(y))
test_size = int(N / 10)
data_ixs = np.random.permutation(np.arange(N))
# train set
train_ix = data_ixs[test_size:]
train_loader = ffnn.create_pytorch_dataset(X[train_ix, :], y[train_ix, :])
# test set
test_ix = data_ixs[:test_size]
test_loader = ffnn.create_pytorch_dataset(X[test_ix, :], y[test_ix, :])

'''NEURAL NETWORK'''
dimensions = train_loader.data.tensors[0].shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ffnn.NeuralNet(input_size=dimensions).to(device)
# train model
epochs, losses, accuracies = ffnn.train(model, train_loader, test_loader, num_epochs=args.num_epochs, device=device)

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
print(metrics)

# Save the model checkpoint
torch.save(model.state_dict(), args.model)
