import argparse
from sklearn.utils import resample
import torch
from scipy import stats
import math
import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-v', '--test_set', required=True)
parser.add_argument('-m', '--model', required=True)  # trained model
parser.add_argument('-b', '--num_boot', type=int, nargs='?', const=10)
args = parser.parse_args()

# read data
X, y = ffnn.load_data(args.test_set)
dimensions = X.shape[1]

# Load saved model and set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ffnn.NeuralNet(input_size=dimensions).to(device)
model.load_state_dict(torch.load(args.model))
model.eval()

# test statistics
test_acc = []  # shape: (num_boot*num_split, )
test_prec = []
test_rec = []
test_f1 = []
test_auc = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for it in range(args.num_boot):

    print(f"Bootstrap iteration {it + 1}")

    X_resample, y_resample = resample(X, y, replace=True)
    test_loader = ffnn.create_data_loader(X_resample, y_resample)

    metrics = ffnn.predict(model, test_loader)
    test_acc.append(metrics['accuracy'])
    test_prec.append(metrics['precision'])
    test_rec.append(metrics['recall'])
    test_f1.append(metrics['F1'])
    test_auc.append(metrics['AUC'])


def summary(values, name=""):
    """
    :param name: name of statistic
    :param values: list of stats
    :return: (name of stat, mean, stderr, conf) of given list
    """
    n, min_max, mean, var, skew, kurt = stats.describe(values)
    stderr = stats.sem(values)
    conf = stats.norm.interval(0.05, loc=mean, scale=math.sqrt(var))
    return '\t'.join([name, str(mean), str(stderr), str(conf)])


# save results
with open("test_stats.tsv", "w") as f:
    f.write('\n'.join([
        summary(test_acc, 'accuracy'),
        summary(test_prec, 'precision'),
        summary(test_rec, 'recall'),
        summary(test_f1, 'F1'),
        summary(test_auc, 'AUROC')
    ]))
