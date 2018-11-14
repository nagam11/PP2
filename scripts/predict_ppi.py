import argparse
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser(description='Predict proteins-protein interactions')
parser.add_argument('-v', '--trained_vectors')
parser.add_argument('-f', '--ppi_fasta')
args = parser.parse_args()

vectors_file = args.trained_vectors
ppi_file = args.ppi_fasta

if (not os.path.isfile(vectors_file) or not os.path.isfile(ppi_file)):
    print("missing input file names")
    exit()

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
features = []
labels = []
for i in range(len(names)):
    seq = seqs[i]
    ppi = ppis[i]
    for j in range(len(seq) - 2 * offset):
        # sequence residues
        seq_vecs = convert_seq_gram(seq[j:j + 2 * offset + 1], vectors, offset)
        features.append(seq_vecs)
        # ppi labels per residue
        ppi_label = max_ppi_label(ppi[j:j + 2 * offset + 1])
        labels.append(ppi_label)

print("features:", len(features), "labels:", len(labels))

# Create model
