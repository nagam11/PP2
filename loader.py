import biovec
import numpy as np
from sklearn.manifold import TSNE

model = biovec.models.load_protvec("trained_models/trained.model")
model.wv.load("trained_models/trained.model")
model.wv.save_word2vec_format(fname="output/trained.vectors")

X = np.loadtxt("output/trained.vectors")
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)
