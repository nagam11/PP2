import argparse
from sklearn.model_selection import StratifiedKFold
import torch
import csv
import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-p', '--ppi_protvecs', required=True)
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', const=50)
parser.add_argument('-k', '--num_split', type=int, nargs='?', const=5)
args = parser.parse_args()


# read data
X, y = ffnn.load_data(args.ppi_protvecs)
dimensions = X.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train data statistics
train_losses = []  # shape: (num_boot*num_split, num_epochs)
val_losses = []
train_acc = []
val_acc = []

kfold = StratifiedKFold(n_splits=args.num_split, shuffle=True)
for i, (train, test) in enumerate(kfold.split(X, y)):
    print('CV-Split {}'.format(i + 1))

    train_loader = ffnn.create_data_loader(X[train], y[train])
    test_loader = ffnn.create_data_loader(X[test], y[test])
    model = ffnn.NeuralNet(input_size=dimensions).to(device)

    # train model
    _, lss, acc = ffnn.train(model, device, train_loader, test_loader,
                             num_epochs=args.num_epochs)

    train_losses.append(lss['train'])
    train_acc.append(acc['train'])
    val_losses.append(lss['validation'])
    val_acc.append(acc['validation'])

with open("epoch_train_losses.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(train_losses)

with open("epoch_train_acc.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(train_acc)

with open("epoch_val_losses.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(val_losses)

with open("epoch_val_acc.tsv", "w") as f:
    csv.writer(f, delimiter='\t').writerows(val_acc)

