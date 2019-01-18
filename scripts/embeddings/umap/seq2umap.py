import biovec
import numpy as np
import umap

import scripts.preprocessing.fasta2vectors as f2v

ppi_file = 'data/ppi_data.fasta'
fasta = f2v.read_fasta(ppi_file)
print(fasta)

aas = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'E':6, 'Q':7, 'G':8, 'H':9, 'I':10, 'L':11, 'K':12, 'M':13,
       'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20}

def sequence2numbers(sequence):
    '''
    :param sequence:
    :return: number representation of sequence as number
    '''

    number_seq = np.empty((len(sequence)), dtype=int)
    for i, x in enumerate(sequence):
        number_seq[i] = aas[x]


print(sequence2numbers(fasta[1]))

