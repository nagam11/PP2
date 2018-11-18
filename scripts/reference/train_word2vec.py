#!/usr/bin/env python

from os import environ
from pathlib import Path
from argparse import ArgumentParser
from gensim.models.word2vec import LineSentence, Word2Vec

from wv_utils import make_corpus, save_w2v_vectors_file


def main():
    arg_parser = ArgumentParser(description='Script to train Word2Vec.')

    arg_parser.add_argument('-i', '--fasta-file')
    arg_parser.add_argument('-o', '--model-file')
    arg_parser.add_argument('-c', '--corpus-file')
    arg_parser.add_argument('-v', '--word-vectors-file')
    arg_parser.add_argument('-u', '--context-vectors-file')

    arg_parser.add_argument('-n', '--ngram-size', type=int, default=3)
    arg_parser.add_argument('-s', '--vector-size', type=int, default=100)
    arg_parser.add_argument('-w', '--window-size', type=int, default=5)
    arg_parser.add_argument('-t', '--num-threads', type=int, default=3)
    arg_parser.add_argument('-r', '--random-seed', type=int, default=None)
    arg_parser.add_argument('-k', '--num-iterations', type=int, default=5)

    args = arg_parser.parse_args()

    fasta_file = args.fasta_file
    model_file = args.model_file
    corpus_file = args.corpus_file
    word_vectors_file = args.word_vectors_file
    context_vectors_file = args.context_vectors_file

    ngram_size = args.ngram_size
    random_seed = args.random_seed
    vector_size = args.vector_size
    window_size = args.window_size
    num_threads = args.num_threads
    num_iterations = args.num_iterations

    if not any([fasta_file, corpus_file]):
        print('Error: Please specify either a FASTA file or corpus file.')
        arg_parser.print_help()
        return

    if fasta_file and not Path(fasta_file).exists():
        print('FASTA file not found: {}'.format(fasta_file))
        return

    if not corpus_file:
        corpus_file = 'corpus.txt'
    elif not fasta_file and not Path(corpus_file).exists():
        print('Corpus file not found: {}'.format(corpus_file))
        return

    if random_seed:
        print('Random-Seed-Mode: Setting number of threads to 1')

        num_threads = 1
        python_hash_seed = environ.get('PYTHONHASHSEED', None)

        if python_hash_seed is None or python_hash_seed == 'random':
            print('Random-Seed-Mode: Global PYTHONHASHSEED needs to be set')
            return
    else:
        random_seed = 42

    if fasta_file:
        make_corpus(fasta_file, corpus_file, ngram_size)

    if not any([model_file, word_vectors_file, context_vectors_file]):
        return

    model = Word2Vec(LineSentence(corpus_file),
                     size=vector_size,
                     window=window_size,
                     min_count=2,
                     sg=1,
                     # hs=0,
                     # negative=5,
                     # ns_exponent=0.75,  # requires gensim 3.5
                     # cbow_mean=1,
                     # sample=0.001,
                     iter=num_iterations,
                     # alpha=0.025,
                     # min_alpha=0.0001,
                     # batch_words=10000,
                     # null_word=0,
                     # trim_rule=None,
                     # compute_loss=False,
                     # sorted_vocab=1,
                     # max_vocab_size=None,
                     # max_final_vocab=None,  # requires gensim 3.5
                     seed=random_seed,
                     workers=num_threads,
                     # callbacks=()
                     )

    if model_file:
        model.save(model_file)

    if word_vectors_file:
        save_w2v_vectors_file(word_vectors_file,
                              model.wv.vocab,
                              model.wv.vectors)

    if context_vectors_file:
        has_syn1 = hasattr(model, 'syn1')  # hierarchical softmax
        has_syn1neg = hasattr(model, 'syn1neg')  # negative sampling

        if has_syn1 and has_syn1neg:
            context_vectors_file_1 = context_vectors_file + '.hs'
            context_vectors_file_2 = context_vectors_file + '.ns'

            save_w2v_vectors_file(context_vectors_file_1,
                                  model.wv.vocab,
                                  model.syn1)
            save_w2v_vectors_file(context_vectors_file_2,
                                  model.wv.vocab,
                                  model.syn1neg)
        elif has_syn1:
            save_w2v_vectors_file(context_vectors_file,
                                  model.wv.vocab,
                                  model.syn1)
        elif has_syn1neg:
            save_w2v_vectors_file(context_vectors_file,
                                  model.wv.vocab,
                                  model.syn1neg)


if __name__ == '__main__':
    main()
