from pathlib import Path


def read_fasta(fasta_file):
    sequences = []

    with Path(fasta_file).open('r') as ff:
        sequence = []

        for line in ff:
            if line.startswith('>'):
                if sequence:
                    sequences.append(''.join(sequence))
                    sequence = []
            else:
                sequence.extend(line.split())

        if sequence:
            sequences.append(''.join(sequence))

    return sequences


def get_n_grams(seq, n):
    n_grams = [[] for i in range(n)]

    for i in range(len(seq) - n + 1):
        n_grams[i % n].append(seq[i:i + n])
    return n_grams


def make_corpus(fasta_file, corpus_file, n):
    with Path(corpus_file).open('w') as cf:
        for sequence in read_fasta(fasta_file):
            for n_grams in get_n_grams(sequence, n):
                cf.write(' '.join(n_grams) + '\n')


def possible_ngrams(alphabet, length, char_list=None, char_index=0):
    if char_list is None:
        char_list = ['.' for i in range(length)]

    if char_index >= length:
        yield ''.join(char_list)
    else:
        for char in alphabet:
            char_list[char_index] = char

            yield from possible_ngrams(alphabet,
                                       length,
                                       char_list,
                                       char_index + 1)


def save_w2v_vectors_file(vectors_file, vocab, vectors):
    sorted_words = list(vocab.keys())

    sorted_words.sort()

    with Path(vectors_file).open('w') as vf:
        vf.write('{} {}\n'.format(vectors.shape[0], vectors.shape[1]))

        for word in sorted_words:
            vector = vectors[vocab[word].index]
            vector = ' '.join(repr(val) for val in vector)
            vf.write('{} {}\n'.format(word, vector))


def save_ft_vectors_file(vectors_file, vectors, min_gram, max_gram, vec_size):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sorted_words = []  # sorted by alphabet order and size

    for n in range(min_gram, max_gram + 1):
        for ngram in possible_ngrams(alphabet, n):
            if ngram in vectors:
                sorted_words.append(ngram)

    with Path(vectors_file).open('w') as vf:
        vf.write('{} {}\n'.format(len(sorted_words), vec_size))

        for word in sorted_words:
            vector = vectors[word]
            vector = ' '.join(repr(val) for val in vector)
            vf.write('{} {}\n'.format(word, vector))
