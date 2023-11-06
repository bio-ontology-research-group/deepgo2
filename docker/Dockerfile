from pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update
RUN apt upgrade -y
RUN apt install -y git-core

RUN git clone https://github.com/bio-ontology-research-group/deepgo2.git
RUN pip install dgl==1.1.2+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

WORKDIR /workspace/deepgo2/

RUN pip install -r requirements.txt

RUN git pull origin main