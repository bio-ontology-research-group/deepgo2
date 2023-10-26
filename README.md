# DeepGO-SE: Protein function prediction as Approximate Semantic Entailment

DeepGO-SE, a novel method which predicts GO functions from protein
sequences using a pretrained large language model combined with a
neuro-symbolic model that exploits GO axioms and performs protein
function prediction as a form of approximate semantic entailment.

This repository contains script which were used to build and train the
DeepGO-SE model together with the scripts for evaluating the model's
performance.

# Dependencies
* The code was developed and tested using python 3.10.
* Clone the repository: `git clone https://github.com/bio-ontology-research-group/deepgo2.git`
* Create virtual environment with Conda or python3-venv module. 
* Install PyTorch: `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`
* Install DGL: `pip install dgl==1.1.2+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html`
* Install other requirements:
  `pip install -r requirements.txt`


# Running DeepGO-SE model (with GOPlus axioms)
Follow these instructions to obtain predictions for your proteins. You'll need
around 30Gb storage and a GPU with >16Gb memory (or you can use CPU)
* Download the [data.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/data.tar.gz)
* Extract `tar xvzf data.tar.gz`
* Run the model `python predict.py -if data/example.fa`


# Training the models
To train the models and reproduce our results:
* Download the [training-data.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/training-data.tar.gz)
* train.py and train_gat.py scripts are used to train different versions of
  DeepGOSE and DeepGOGATSE models correspondingly
* train_cnn.py, train_mlp.py and train_dgg.py scripts are used to train
  baseline models DeepGOCNN, MLP and DeepGraphGO.
* Examples:
  - Train a single DeepGOZero MFO prediction model which uses InterPRO annotation features \
    `python train.py -m deepgozero -ont mf`
  - Train a single DeepGOZero CCO prediction model which uses ESM2 embeddings \
    `python train.py -m deepgozero_esm -ont cc`
  - Train a single DeepGOGAT BPO prediction model which uses predicted MF features \
    `python train.py -m deepgogat_mfpreds_plus -ont bp`
    
    
## Evaluating the predictions
The training scripts generate predictions for the test data that are used
to compute evaluation metrics.
* To evaluate single predictions run evaluate.py script. Example: \
  `python evaluate.py -m mlp -td test -on mf`
* To evaluate approximate entailment predictions use evaluate_entailment.py
  script. Example: \
  `python evaluate_entailment.py -m deepgozero_esm -td nextprot -ont cc` \
  Note: this script requires multiple trained models with performance reports
  on the validation set.
# Citation

If you use DeepGO-SE for your research, or incorporate our learning
algorithms in your work, please cite: Maxat Kulmanov, Francisco
J. Guzman-Vega, Paula Duek, Lydie Lane, Stefan T. Arold, Robert
Hoehndorf; DeepGO-SE: Protein function prediction as Approximate
Semantic Entailment
