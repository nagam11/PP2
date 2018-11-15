import biovec

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score
from json_tricks import dumps

model = biovec.models.load_protvec("trained_models/trained.model")
model.wv.load_word2vec_format(fname="output/trained.vectors")


def read_fasta(path):
    """
    Reads amino sequences and properties from fasta-formatted file

    :param path: path to fasta file
    :return: list of dicts fo fasta sequences and binding properties {'seq':, 'bind':}
    """
    fasta = []

    with open(path, 'r') as f:
        # Intermediary variables for the current header and sequence
        header = ''
        amino_sequence = ''
        binding = ''
        # Start reading the input file line by line
        while True:
            # Read the line
            line = f.readline()

            # If the header is not empty,
            if header:
                # Check if the current line is empty or contains no whitespace chars
                # If so, reached the end of sequence - add the header and sequence tuple to the list
                # and reset the intermediary variables
                if line.strip():
                    if not (line.startswith('>') or line.startswith(';')):
                        if not (line.startswith('-') or line.startswith('+')):
                            # If the line is not empty and is not a set of non-printing chars, read the sequence
                            amino_sequence = line.strip()
                        else:
                            binding = line.strip()
                    else:
                        fasta.append({'seq': amino_sequence, 'bind': binding})
                        header = ''
                        amino_sequence = ''
                        binding = ''

            # If the header variable is empty, check if the current line is the header
            # In the case, initialise the header with the line and begin a new loop iteration
            if line.startswith('>') or line.startswith(';'):
                if not amino_sequence:
                    header = line.strip()

            # End Of File reached - break the loop
            if line == '':
                fasta.append({'seq': amino_sequence, 'bind': binding})
                break
    return fasta


def compute_vector(word):
    """
    Compute word vector by sum over triples

    :param word: string of amino residues
    :return: element-wise sum of vectors as ndarray(model.wv.vector_size)
    """
    return sum([model.wv.get_vector(x) for x in [word[i:i + 3] for i in range(len(word) - 2)]])


def compose_data(path):
    """
    Put the inputs (word vectors) and targets together

    :param path: path to fasta file
    :return: ndarray(n_samples, n_features), ndarray(n_samples,)
    """
    fasta = read_fasta(path)
    vectors = []
    bindings = []

    for entry in fasta:
        seq = entry.get('seq')
        vectors.extend([compute_vector(seq[i - 3:i + 4]) for i in range(3, len(seq) - 3)])
        bindings.append(np.array([1 if entry.get('bind')[i] == '+' else 0 for i in range(3, len(seq) - 3)]))

    return np.stack(vectors, axis=0), np.hstack(bindings)


# Get the data
X, y = compose_data('data/ppi_data.fasta')

# Initialise k-fold validator
skf = StratifiedKFold(n_splits=10, random_state=42)

# Arrange scoring functions
scoring = {'Accuracy': make_scorer(accuracy_score),
           'Precision': make_scorer(precision_score),
           'Recall': make_scorer(recall_score),
           'AUC': make_scorer(roc_auc_score)}

# Set the parameters
parameters = {'max_iter': [100, 200, 500], 'hidden_layer_sizes': [(25,), (50,), (100,)],
              'random_state': [42], 'learning_rate': ['constant'], 'learning_rate_init': [0.001, 0.01]}

# Initialise grid search for hyperparameters with MLPClassifier
clf = GridSearchCV(MLPClassifier(), parameters, cv=skf, scoring=scoring, refit='AUC', return_train_score=True)
clf.fit(X, y)

result = dumps(clf.cv_results_)
print(result)

with open('result.json', 'w') as f:
    f.write(result)
