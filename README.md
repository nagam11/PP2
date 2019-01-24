# PP2 Exercise

We provide two different prediction models.

1. Word2Vec Embeddings with a NN in PyTorch
2. [Google Seq2Seq Prediction](https://github.com/nagam11/PP2/tree/master/scripts/s2s)

### Running

To run the project, you need to use Python 3.6+. The required dependencies are listed below: 

* torch 
* torchvision
* numpy 
* matplotlib
* biovec 
* sklearn 

To install the dependencies run

        pip install -r requirements.txt 

## PyTorch Prediction
To run the protein-protein binding predictions 

        python predict.py -i input_file.fasta -o output_file.fasta
        
Example input and output files can be found under: 
`data/test_input.fasta ` and `data/test_output.fasta`


       
 
        
