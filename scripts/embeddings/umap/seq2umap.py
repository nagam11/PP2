import numpy as np
import umap
from matplotlib import pyplot as plt

import scripts.preprocessing.utils as pp

ppi_file = '../../../data/ppi_data.fasta'
fasta = pp.read_fasta(ppi_file)

aas = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13,
       'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}


def max_length(sequences):
    """
    :param sequences: contents of fasta file as dictionary
    :return: length of longest sequence in fasta
    """
    m = 0
    for seq in sequences:
        if len(seq) > m:
            m = len(seq)
    return m


def sequence2numbers(sequence, max_length):
    """
    :param max_length: length of output vectors (length of longest sequence)
    :param sequence: protein sequence
    :return: numpy array of max_length  with number representation of sequence as number
    """
    number_seq = np.zeros(max_length, dtype=float)
    for i, x in enumerate(sequence):
        number_seq[i] = aas[x] if x in aas else 0
    return number_seq


def bind2numbers(binding):
    """
    :param binding: string of binding positions
    :return: translation into 1 (+) and 0 (-)
    """
    n = np.zeros(len(binding))
    for i, x in enumerate(binding):
        if x == '+':
            n[i] = 1
    return n


# protein sequences
seqs = [s['seq'] for s in fasta]
m_len = max_length(seqs)
seq_vecs = np.array([sequence2numbers(s, m_len) for s in seqs])
embedded = umap.UMAP().fit_transform(seq_vecs)

# TODO: figure out how to add labels
# binding sequences
# bindings = [bind2numbers(s['bind']) for s in fasta]
# np.append(embedded, np.vstack(bindings), axis=1)

np.save("padded_vectors.npy", seq_vecs)
np.save("padded_vectors_umap.npy", embedded)

plt.scatter(embedded[:,0], embedded[:,1])
plt.savefig('padded_umap.png')


