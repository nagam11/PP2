import numpy as np
import umap


ppi_as_vec = np.load("../../preprocessing/ppi_as_vec.npy")
labels = np.vstack(ppi_as_vec[:, -1])
ppi_umap = umap.UMAP().fit_transform(ppi_as_vec[:, :-1])

data = np.append(ppi_umap, labels, axis=1)
np.save("ppi_as_w2v_umap.npy", data)
print("Done.")
