import argparse
import biovec
from sklearn.neural_network import MLPClassifier


parser = argparse.ArgumentParser(description='Predict proteins-protein interactions')
parser.add_argument('-i', '--input_fasta_file')
parser.add_argument('-o', '--output_file')
args = parser.parse_args()

# read the data
#  1. vectors
#  2. PPI input
# translate protein sequence into feature vectors
#  1. extract each residue and surrounding of sliding window
#  2. translate single residue into 7-gram -> 3-gram
gram_7 = "VWLNGEP"

#  3. get vector for 3-grams
# get labels of data points

MLPClassifier(hidden_layer_sizes=(25, 50, 100),  
              learning_rate_init=(0.001, 0.01),
              n_iter_no_change=(100, 200, 500))
