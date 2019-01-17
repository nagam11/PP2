import argparse
import os
import tensorflow as tf
import numpy as np

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

# Configure GPU not to use all memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Start a new session and initialize variables
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

# TODO: set up network Prediction
#score =
#predictions = tf.nn.softmax()
#predictions = tf.argmax(predictions, 1)

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

        # TODO Run predictions
        # Create ngrams from sequence for inputs
        '''X, _ = 
        for j in range(len(X)):
            score, prediction = sess.run([scores, predictions], feed_dict={inputs: X_test[i, None]})
            if prediction == 1:
                pred = '+'
            else:
                pred = '-'
            # Write predicted bindings
            file.write("" + "\t" + pred + "\t" + round(max(score), 2) + "\n")
        '''

        for a in range(3, len(protein_sequences[i]) - 3):
            file.write(protein_sequences[i][a] + "\t" + "" + "\t" + "" + "\n")

        # Note: 3-gram model does not allow us to predict the last 3 residues
        # Mark these residues as n/a
        for j in range(-3, 0):
            file.write(protein_sequences[i][j:] + "\tn/a\tn/a\n" if j == -1
                       else protein_sequences[i][j:j+1] + "\tn/a\tn/a\n")


print("\nPrediction has finished. Please check output files.")