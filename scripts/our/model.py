import datetime

import biovec

import numpy as np
from json_tricks import dumps
from scripts.reference.ml_sample_code import train_and_optimize


class Model(object):

    fasta = None
    trained_vectors = {}
    layer_sizes = (50, )

    def __init__(self, fasta_path, trained_vectors, layer_sizes):
        self.fasta = self.read_fasta(fasta_path)

        for key, vectors in trained_vectors.items():
            self.trained_vectors.update({key: vectors})

        if layer_sizes is not None:
            self.layer_sizes = layer_sizes

    @staticmethod
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

    def compute_vector(self, vectors, word):
        """
        Compute word vector by sum over triples

        :param vectors:
        :param word: string of amino residues
        :return: element-wise sum of vectors as ndarray(model.wv.vector_size)
        """
        return sum([vectors.wv.get_vector(x) for x in [word[i:i + 3] for i in range(len(word) - 2)]])

    def compose_data(self, vectors):
        """
        Put the inputs (word vectors) and targets together

        :param path: path to fasta file
        :return: ndarray(n_samples, n_features), ndarray(n_samples,)
        """
        vecs = []
        bindings = []

        for entry in self.fasta:
            seq = entry.get('seq')
            vecs.extend([self.compute_vector(vectors, seq[i - 2:i + 3]) for i in range(2, len(seq) - 2)])
            bindings.append(np.array([1 if entry.get('bind')[i] == '+' else 0 for i in range(2, len(seq) - 2)]))
            vecs.extend([self.compute_vector(vectors, seq[i - 3:i + 4]) for i in range(3, len(seq) - 3)])
            bindings.append(np.array([1 if entry.get('bind')[i] == '+' else 0 for i in range(3, len(seq) - 3)]))
            vecs.extend([self.compute_vector(vectors, seq[i - 4:i + 5]) for i in range(4, len(seq) - 4)])
            bindings.append(np.array([1 if entry.get('bind')[i] == '+' else 0 for i in range(4, len(seq) - 4)]))

        return np.stack(vecs, axis=0), np.hstack(bindings)

    def train(self):
        print('Training started')
        print('Vector sizes: ' + ', '.join(self.trained_vectors.keys()))
        print('Layer sizes: ' + str(self.layer_sizes))
        for key, vectors in self.trained_vectors.items():
            print('Training for vector size: ' + key)
            # Get the data
            x, y = self.compose_data(vectors)

            print(str(datetime.datetime.now()))
            cv_results, refined_result = train_and_optimize(x, y, self.layer_sizes)
            print(str(datetime.datetime.now()))

            for i in range(len(cv_results)):
                with open('cv_' + key + '_outersplit_' + str(i) + '_' + str(self.layer_sizes) + '.json', 'w') as f:
                    f.write(dumps(cv_results[i]))

            with open('refined_result_' + key + '_' + str(self.layer_sizes) + '.txt', 'w') as f:
                f.write(refined_result)


tr_vecs = {}
for i in ['100']:
    model = biovec.models.load_protvec("../../trained_models/trained" + i + ".model")
    tr_vecs.update({i: model.wv.load_word2vec_format(fname="../../output/trained" + i + ".vectors")})

for j in [(50, 25, 50, 25, 50, 25), (50, 50, 50, 50, 50, 50), (125, 100, 75, 50, 25, 5), (5, 25, 50, 75, 100, 125)]:
    model = Model('../../data/ppi_data.fasta', tr_vecs, j)
    model.train()
