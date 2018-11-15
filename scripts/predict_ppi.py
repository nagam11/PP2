import argparse
import os
import timeit
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score

parser = argparse.ArgumentParser(description='Predict proteins-protein interactions')
parser.add_argument('-v', '--trained_vectors')
parser.add_argument('-f', '--ppi_fasta')
parser.add_argument('-o', '--opti_output')
parser.add_argument('-j', '--jobs', default=1, type=int)
args = parser.parse_args()

vectors_file = args.trained_vectors
ppi_file = args.ppi_fasta

if (not os.path.isfile(vectors_file) or not os.path.isfile(ppi_file)):
    print("missing input file names")
    exit()

start = timeit.default_timer()

# read the data
#  1. vectors
vectors = {}
with open(vectors_file) as file:
    i = 0
    for line in file:
        aa_vec = line.split()
        feature = np.array([float(x) for x in aa_vec[1:]])
        vectors.update({aa_vec[0]: feature})
        i = i + 1
print("trained vectors:", len(vectors))

#  2. parse PPI input
names = []
seqs = []
ppis = []
# read data
with open(ppi_file) as file:
    i = 0
    for line in file:
        if (i % 3 == 0):
            # name
            names.append(line[1:].strip())
        elif (i % 3 == 1):
            # sequence
            seqs.append(line.strip())
        elif (i % 3 == 2):
            # labels
            ppis.append(line.strip())
        i = i + 1

print("sequences:", len(seqs))
print("time for reading:", timeit.default_timer() - start)
start = timeit.default_timer()

# translate protein sequence into feature vectors
def convert_seq_gram(long_gram, vectors, offset=3):
    seq_gram_strings = [long_gram[i:i + offset] for i in range(2 * offset - 1)]  # get short grams within long gram
    seq_gram_vectors = [vectors[gram] for gram in seq_gram_strings]  # convert into feature vectors
    gram_sum = np.array(seq_gram_vectors).sum(axis=0)  # sum up short grams
    return gram_sum


def max_ppi_label(ppi_string):
    ppi_vector = list(map(lambda x: 0 if x == "-" else 1, ppi_string))
    labels = {x: list(ppi_vector).count(x) for x in set(ppi_vector)}
    return max(labels, key=labels.get)


offset = 3
X = []
y = []
for i in range(len(names)):
    seq = seqs[i]
    ppi = ppis[i]
    for j in range(len(seq) - 2 * offset):
        # sequence residues
        seq_vecs = convert_seq_gram(seq[j:j + 2 * offset + 1], vectors, offset)
        X.append(seq_vecs)
        # ppi labels per residue
        ppi_label = max_ppi_label(ppi[j:j + 2 * offset + 1])
        y.append(ppi_label)

print("features:", len(X), "labels:", len(y))

# Create model
parameter_space = {
    'hidden_layer_sizes': [25, 50, 100],
    'learning_rate_init': [0.001, 0.01],
    'n_iter_no_change': [100, 200, 500],
}
mlp = MLPClassifier(random_state=42)
skf = StratifiedKFold(n_splits=10, random_state=42)
scoring = {'Accuracy': make_scorer(accuracy_score),
           'Precision': make_scorer(precision_score),
           'Recall': make_scorer(recall_score),
           'AUC': make_scorer(roc_auc_score)}
clf = GridSearchCV(mlp, parameter_space, n_jobs=args.jobs, cv=skf,
                   scoring=scoring, refit='AUC', return_train_score=True).fit(X, y)

opt_time = timeit.default_timer() - start
print("time:", opt_time)
print('best parameters found:\n', clf.best_params_)

with open(args.opti_output, "w+") as f:
    f.write("time:", opt_time)
    f.write('best parameters found:\n', clf.best_params_)
    f.write(str(clf.cv_results_))
