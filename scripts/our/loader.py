import biovec
import numpy as np
from sklearn.manifold import TSNE

model = biovec.models.load_protvec("../../trained_models/trained200.model")
model.wv.load("../../trained_models/trained200.model")
model.wv.save_word2vec_format(fname="../../output/trained200.vectors")
