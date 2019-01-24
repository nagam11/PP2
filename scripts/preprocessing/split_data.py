import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='Split training and test sets')
parser.add_argument('-p', '--ppi_protvecs', required=True)
parser.add_argument('-t', '--train_set', required=True)
parser.add_argument('-v', '--test_set', required=True)
args = parser.parse_args()

# read data
X, y = ffnn.load_data(args.ppi_protvecs)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
# train_set = np.append(X_train, np.vstack(y_train), axis=1)
# test_set = np.append(X_test, np.vstack(y_test), axis=1)

N = int(len(y))
test_size = int(N / 10)
data_ixs = np.random.permutation(np.arange(N))
# train set
train = data_ixs[test_size:]
test = data_ixs[:test_size]

train_set = np.append(X[train], y[train])
test_set = np.append(X[test], y[test])

np.save(args.train_set, train_set)
np.save(args.test_set, test_set)
