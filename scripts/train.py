# -*- coding: utf-8 -*-

import argparse
import biovec


parser = argparse.ArgumentParser(description='This script is ...')
parser.add_argument('-i', '--input_fasta_file')
parser.add_argument('-o', '--output_file')
args = parser.parse_args()

# n = 3 (word length; 3-gram) MAF,SAE etc.
# size = 100 (size of feature-vector), every 3-gram is represented as a vector of size 100
# window = 5 (max. distance to neighbouring words)
# sg = 1 (skip-gram algorithm)
# min_count = 2 (min. number a k-gram has to be in the corpus)
model = biovec.ProtVec(corpus_fname=args.input_fasta_file, n=3, size=200, window=5, sg=1, min_count=2, workers=4)
model.save(args.output_file)
print("Model is ready")