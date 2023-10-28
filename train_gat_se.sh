#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'Please provide model name and ontology'
    exit 1
fi

for i in `seq 0 9`; do
    echo $i
    echo Training model $1_$i for ontology $2
    python train_gat.py -m $1 -mi $i -ont $2
done
