import biovec
import numpy as np

import scripts.preprocessing.utils as pp

model = biovec.models.load_protvec("../../trained_models/trained.model")
model.wv.load_word2vec_format(fname="../../output/trained.vectors")

# Get the data
X, y = pp.compose_data('../../data/ppi_data.fasta', model)
data = np.append(X, np.vstack(y), axis=1)
np.save("ppi_as_vec.npy", data)

print('test loading')
load_data = np.load("ppi_as_vec.npy")
print(load_data)
print(" ----FINISHED----")
