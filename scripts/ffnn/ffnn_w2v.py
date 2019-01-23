import argparse
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import torch
import numpy as np

import scripts.ffnn.model as ffnn

parser = argparse.ArgumentParser(description='FFNN models')
parser.add_argument('-p', '--ppi_protvecs', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-e', 'num_epochs', required=True)
args = parser.parse_args()


train_dataset, test_dataset = ffnn.load_and_split(args.ppi_protvecs)
'''# Sampler for DataLoader
#class_sample_counts = [65365, 24176]
class_sample_counts = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
samples_weights = weights[y_train]
sampler = torch.utils.data.sampler.WeightedRandomSampler(
                                weights=samples_weights,
                                num_samples=len(samples_weights),
                                replacement=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler=sampler)                                
'''
batch_size = 100
dimensions = train_dataset.tensors[0].shape[1]
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# loader redundant?
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ffnn.NeuralNet(input_size=dimensions).to(device)

epochs, losses, accuracies = ffnn.train(model, train_loader, test_loader, num_epochs=args.num_epochs, device=device)

# plot losses
training, = plt.plot(epochs, losses['train'], label='training')
validation, = plt.plot(epochs, losses['validation'], label='validation')
plt.legend(handles=[training, validation])
plt.title("Loss over epochs for FFNN")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# plot accuracy
training, = plt.plot(epochs, accuracies['train'], label='training')
validation, = plt.plot(epochs, accuracies['validation'], label='validation')
plt.legend(handles=[training, validation])
plt.title("Accuracy over epochs for FFNN on Protvec")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

metrics = ffnn.predict(model, train_loader)
print(metrics)

# Save the model checkpoint
torch.save(model.state_dict(), args.model)
