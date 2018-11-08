# -*- coding: utf-8 -*-

import argparse
import biovec


parser = argparse.ArgumentParser(description='This script is ...')
parser.add_argument('-i', '--input_fasta_file')
parser.add_argument('-o', '--output_file')
args = parser.parse_args()

model = biovec.ProtVec(corpus_fname=args.input_fasta_file, n=3, size=100, window=5, sg=1, min_count=2, workers=2)
model.save(args.output_file)
