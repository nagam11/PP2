import argparse
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from scipy import stats
import math
import csv

import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-p', '--ppi_protvecs', required=True)
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', const=50)
parser.add_argument('-b', '--num_boot', type=int, nargs='?', const=10)
parser.add_argument('-k', '--num_split', type=int, nargs='?', const=5)
args = parser.parse_args()

# read data
X, y = ffnn.load_data(args.ppi_protvecs)

# set parameters of bootstrap
n = X.shape[0]
test_size = int(n / 10)
data_ix = np.arange(n)

# train data statistics
train_losses = []  # shape: (num_boot*num_split, num_epochs)
val_losses = []
train_acc = []
val_acc = []

# test statistics
test_acc = []  # shape: (num_boot*num_split, )
test_prec = []
test_rec = []
test_f1 = []
test_auc = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for it in range(args.num_boot):

    print(f"Bootstrap iteration {it + 1}")
    # resample bootstrap data
    boot = resample(data_ix, replace=True, n_samples=n - test_size)
    # holdout
    test_ix = [i for i in data_ix if i not in boot]
    test_loader = ffnn.create_data_loader(X[test_ix, :], y[test_ix])
    dimensions = X.shape[1]

    kfold = StratifiedKFold(n_splits=args.num_split, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(X[boot, :], y[boot, :])):
        print('CV-Split {}'.format(i + 1))

        train_loader = ffnn.create_data_loader(X[train], y[train])
        test_loader = ffnn.create_data_loader(X[test], y[test])
        model = ffnn.NeuralNet(input_size=dimensions).to(device)

        # train model
        _, lss, acc = ffnn.train(model, train_loader, test_loader,
                                 num_epochs=args.num_epochs, device=device)

        train_losses.append(lss['train'])
        train_acc.append(acc['train'])
        val_losses.append(lss['validation'])
        val_acc.append(acc['validation'])

        # test
        metrics = ffnn.predict(model, test_loader)
        test_acc.append(metrics['accuracy'])
        test_prec.append(metrics['precision'])
        test_rec.append(metrics['recall'])
        test_f1.append(metrics['F1'])
        test_auc.append(metrics['AUC'])


# compute test statistics
def summary(values):
    """
    :param values: list of stats
    :return: (name of stat, mean, stderr, conf) of given list
    """
    n, min_max, mean, var, skew, kurt = stats.describe(values)
    stderr = stats.sem(values)
    conf = stats.norm.interval(0.05, loc=mean, scale=math.sqrt(var))
    return '\t'.join(['test accuracy', mean, stderr, conf])


# save results
with open("test_stats.tsv", "w") as f:
    f.writelines([
        summary(test_acc),
        summary(test_prec),
        summary(test_rec),
        summary(test_f1),
        summary(test_auc)
    ])

with open("epoch_train_losses.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(train_losses)

with open("epoch_train_acc.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(train_acc)

with open("epoch_val_losses.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(val_losses)

with open("epoch_val_acc.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(val_acc)
