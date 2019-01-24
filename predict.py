import torch
import argparse
import os
import numpy as np
import biovec
import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='Predict ppi bindings')
parser.add_argument('-i', '--input_file', required=True)
parser.add_argument('-o', '--output_file', required=True)
args = parser.parse_args()

# Read files from user
input_file = args.input_file
output_file = args.output_file

# Check that files are not missing
if not os.path.isfile(input_file) or not os.path.isfile(output_file):
    print("missing files")
    exit()

# Prepare word2vec embeddings from user input
modeL = biovec.models.load_protvec("trained_models/trained.model")
modeL.wv.load_word2vec_format(fname="output/trained.vectors")


def compute_vector(word):
    return sum([modeL.wv.get_vector(x) for x in [word[i:i + 3] for i in range(len(word) - 2)]])


def compose_data(seq):
    vectors = []
    vectors.extend([compute_vector(seq[i - 3:i + 4]) for i in range(3, len(seq) - 3)])
    return np.stack(vectors, axis=0)


# Collect protein names and amino-acids sequences
proteins_names = []
protein_sequences = []
with open(input_file) as file:
    i = 0
    for line in file:
        line = line.strip()
        if i % 2 == 0:
            proteins_names.append(line)
        else:
            protein_sequences.append(line)
        i = i + 1


# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load NN and use same parameters as for the trained model
dimensions = compose_data(protein_sequences[0]).shape[1]
model = ffnn.NeuralNet(input_size=dimensions).to(device)

# Load saved model and set to evaluation mode
model.load_state_dict(torch.load("scripts/ffnn/ffnn_model.ckpt"))
model.eval()

# Write to output file
with open(output_file, "w") as file:
    for i in range(len(proteins_names)):

        # Write protein name
        file.write(proteins_names[i] + "\n")

        # Note: 3-gram model does not allow us to predict the first 3 residues
        # Mark these residues as n/a
        j = 0
        for j in range(3):
            file.write(protein_sequences[i][j] + "\tn/a\tn/a\n")

        # Create n-grams from sequence for inputs
        X_test = compose_data(protein_sequences[i])

        # Predict on new sequences
        prediction = model(torch.from_numpy(np.float32(X_test)))
        _, predicted = torch.max(prediction, 1)
        predicted_bindings = predicted.numpy()

        # Get prediction probabilities
        sm = torch.nn.Softmax(dim=1)
        probabilities, _ = torch.max(sm(prediction), 1)
        prediction_probabilities = probabilities.detach().numpy()

        # Parse predicted bindings and the probability
        for a in range(3, len(protein_sequences[i]) - 3):
            if predicted_bindings[a - 3] == 0:
                binding = '-'
            else:
                binding = '+'
            file.write(protein_sequences[i][a] + "\t" + str(binding) + "\t" + str(round(prediction_probabilities[a - 3],3)) + "\n")

        # Note: 3-gram model does not allow us to predict the last 3 residues
        # Mark these residues as n/a
        for j in range(-3, 0):
            file.write(protein_sequences[i][j:] + "\tn/a\tn/a\n" if j == -1
                       else protein_sequences[i][j:j+1] + "\tn/a\tn/a\n")


print("\nPrediction has finished. Please check output files.")

