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
>30Gb storage and GPU with >16Gb memory (or you can use CPU)
* Download the [data.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/data.tar.gz)
* Extract `tar xvzf data.tar.gz`
* Run the model `python predict.py -if data/example.fa`


# Training the models
The scripts require GeneOntology in OBO and OWL Formats.

## Citation

If you use DeepGO-SE for your research, or incorporate our learning
algorithms in your work, please cite: Maxat Kulmanov, Francisco
J. Guzman-Vega, Paula Duek, Lydie Lane, Stefan T. Arold, Robert
Hoehndorf; DeepGO-SE: Protein function prediction as Approximate
Semantic Entailment
