# PP2 Exercise

We provide two different prediction models.

1. Word2Vec Embeddings with a feed-forward neural network in PyTorch
2. [Google Seq2Seq Prediction](https://github.com/nagam11/PP2/tree/master/scripts/s2s)

### Running

To run the project, you need to use Python 3.6+. The required dependencies are listed below: 

```{python}
torch 
torchvision
numpy 
matplotlib
biovec 
sklearn 
```

To install the dependencies run

        pip install -r requirements.txt 

## Feed-Forward Neural Network
### Run Prediction
To run the protein-protein binding predictions 

        python predict.py -i input_file.fasta -o output_file.fasta
        
Example input and output files can be found under: `data/test_input.fasta` and `data/test_output.fasta`. The script uses a pre-trained model that can be found under `trained_models/ffnn_model.ckpt`

### Parse and Split the Data
The data needs to be split into a training and test set for later cross-valdiation and bootstrapping.
```{bash}
python scripts/preprocessing/split_data.py \
		--ppi_protvecs=scripts/preprocessing/ppi_as_vec.npy \
		--train_set=ppi_vec_train.npy \
		--test_set=ppi_vec_test.npy
```

### Train Model
```{bash}
python scripts/ffnn/train_ffnn_w2v.py \
		--training_set=scripts/preprocessing/ppi_vec_train.npy \
		--model=trained_models/ffnn_model.ckpt \
		--num_epochs=100 \
		--batch_size=100
```

### Validation
```{bash}
python scripts/ffnn/validation/cross_validation.py \
		--ppi_protvecs=scripts/preprocessing/ppi_vec_train.npy \
		--num_epochs=100 \
		--num_split=5
```

### Bootstrapping
```{bash}
python scripts/ffnn/validation/bootstrapping.py \
		--test_set=scripts/preprocessing/ppi_vec_test.npy \
		--model=trained_models/ffnn_model.ckpt \
		--num_boot=1000
```
### Visualistation
The results of cross-validation and bootstrapping are summarised in `scripts/postprocessing/final_model.html`
