import argparse
from sklearn.utils import resample
import numpy as np
import torch
from scipy.stats import sem

import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-p', '--ppi_protvecs', required=True)
parser.add_argument('-e', '--num_epochs', type=int, required=True)
args = parser.parse_args()

# read data
X, y = ffnn.load_data(args.ppi_protvecs)

# set parameters of bootstrap
n = X.shape[0]
test_size = int(n / 10)
data_ix = np.arange(n)
num_boot = 10  # number of bootstrap iterations

accuracies = []
for it in range(5):
    boot = resample(data_ix, replace=True, n_samples=n - test_size)
    train_loader = ffnn.create_pytorch_dataset(X[boot, :], y[boot, :])
    test_ix = [i for i in data_ix if i not in boot]
    test_loader = ffnn.create_pytorch_dataset(X[test_ix, :], y[test_ix])

    dimensions = train_loader.dataset.tensors[0].shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ffnn.NeuralNet(input_size=dimensions).to(device)
    # train model
    ffnn.train(model, train_loader, test_loader,
                                  num_epochs=args.num_epochs,
                                  device=device)
    metrics = ffnn.predict(model, test_loader)
    accuracies.append(metrics['accuracy'])

print(np.mean(np.array(accuracies)))
print(sem(accuracies))
