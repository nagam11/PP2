import argparse
import biovec
from sklearn.neural_network import MLPClassifier


parser = argparse.ArgumentParser(description='Predict proteins-protein interactions')
parser.add_argument('-i', '--input_fasta_file')
parser.add_argument('-o', '--output_file')
args = parser.parse_args()

random_state = 42


MLPClassifier(hidden_layer_sizes=(25, 50, 100),  
              learning_rate_init=(0.001, 0.01),
              n_iter_no_change=(100, 200, 500))
