# DeepGO-SE: Protein function prediction as Approximate Semantic Entailment

DeepGO-SE, a novel method which predicts GO functions from protein
sequences using a pretrained large language model combined with a
neuro-symbolic model that exploits GO axioms and performs protein
function prediction as a form of approximate semantic entailment.

This repository contains script which were used to build and train the
DeepGO-SE model together with the scripts for evaluating the model's
performance.

## Dependencies
* The code was developed and tested using python 3.10.
* Install PyTorch: `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`
* Install DGL: `pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html`
* Install other requirements:
  `pip install -r requirements.txt`


## Data
* https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/ - Here you can find the data
used to train and evaluate our method.
 * data.tar.gz - UniProtKB-SwissProt dataset (release 21_04)
 * data-nextprot.tar.gz - neXtProt dataset

## Scripts
The scripts require GeneOntology in OBO and OWL Formats.
* uni2pandas.py - This script is used to convert data from UniProt
database format to pandas dataframe.
* deepgo2_data.py - This script is used to generate training and
  testing datasets.
* deepgozero_esm.py - This script is used to train the model with GO axioms
* deepgozero_esm_plus.py - This script is used to train the model with GOPlus axioms
* deepgozero_gat.py - This script is used to train the model which combines Graph Attention Networks
* deepgozero_gat_plus.py - This script is used to train the GAT model with GOPlus axioms
* deepgozero_gat_mf.py - This script is used to train the the GAT model with molecular functions as features
* deepgozero_gat_mf_plus.py - This script is used to train the GAT model with MF features and GOPlus axioms
* deepgozero_gat_mfpreds.py - This script is used to train the GAT model with predicted MF features
* deepgozero_gat_mfpreds_plus.py - This script is used to train the GAT model with predicted MF features and GOPlus axioms
* evaluate.py - The scripts are used to compute Fmax, Smin and AUPR
* evaluate_terms.py - The scripts are used to compute class-centric average AUC
* evaluate_entailment.py - The scripts are used to evaluate semantic entailment
* Normalizer.groovy - The script used to normalize OWL ontology
* Corpus.groovy - This script is used to extract class axiom definitions
* deepgopro.py - This script is used to train the MLP baseline model
* deepgocnn.py - This script is used to train the DeepGOCNN model



## Citation

If you use DeepGO-SE for your research, or incorporate our learning
algorithms in your work, please cite: Maxat Kulmanov, Francisco
J. Guzman-Vega, Paula Duek, Lydie Lane, Stefan T. Arold, Robert
Hoehndorf; DeepGO-SE: Protein function prediction as Approximate
Semantic Entailment
