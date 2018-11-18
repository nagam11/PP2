import biovec
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Read PPI data from ppi_data.fasta file
protein_names, protein_sequences, protein_labels = [], [], []

with open('../data/ppi_data.fasta') as file:
    lines = file.read().splitlines()

    for i in range(len(lines)):

        # split every three lines since each protein is represented by three lines
        if i % 3 == 0:
            protein_names.append(lines[i])
        elif i % 3 == 1:
            protein_sequences.append(lines[i])
        elif i % 3 == 2:
            '''
                Create an array for each protein containing information about bindings. 
                'is-binding' : 1, 'non-binding' : 0
            '''
            lb = []
            for letter in lines[i]:
                if letter == "+":
                    label = 1
                elif letter == "-":
                    label = 0
                lb.append(label)
            protein_labels.append(lb)

# ML Pipeline with stratified 10-fold cross-validation
window_size = 7
vector_size = 100
X, y = [], []
# Use trained model from exercise 1 with SwissProt dataset
pv = biovec.models.load_protvec('../trained_models/trained.model')

# Build vectors
for i in range(len(protein_sequences)):

    for j in range(0, len(protein_sequences[i]) - window_size + 1):

        sub_sequence = protein_sequences[i][j:j + window_size]

        # Sum over all 3-grams within window
        tmp = np.zeros(vector_size)
        for k in range(0, window_size - 2):
            tmp = tmp + np.array(pv[sub_sequence[k:k + 3]])

        X.append(tmp)
        y.append(protein_labels[i][j + 3])

X, y = np.array(X), np.array(y)

skf = StratifiedKFold(n_splits=10, random_state=42)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # TODO Implement GridSearchCV
    mlp = MLPClassifier(hidden_layer_sizes=(100,),
                        learning_rate_init=0.001,
                        max_iter=1)

    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    print("Accuracy: %a" % accuracy_score(y_test, predictions))
    print("Precision: %a" % precision_score(y_test, predictions))
    print("Recall %a " % recall_score(y_test, predictions, average='micro'))
    print("AUC/ROC: %a" % roc_auc_score(y_test, predictions))

    break


