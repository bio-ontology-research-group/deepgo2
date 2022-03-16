#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from jpype import *
import jpype.imports
import os
import pickle as pkl

logging.basicConfig(level=logging.INFO)

jars_dir = "gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir+name for name in os.listdir(jars_dir)])}'
startJVM(getDefaultJVMPath(), "-ea",  "-Djava.class.path=" + jars,  convertStrings=False)


from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor
from org.semanticweb.owlapi.reasoner import SimpleConfiguration
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.util import InferredClassAssertionAxiomGenerator
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager


# current directory is the mowl project root
# example parameters: --ont-file=../data/ppi_yeast_localtest/goslim_yeast.owl --data-file=../data/ppi_yeast_localtest/4932.protein.physical.links.v11.5.txt.gz --go-annots-file=../data/ppi_yeast_localtest/sgd.gaf.short --out-dir=../data/ppi_yeast_localtest


DATAROOT = "data/"
FOLDERS = ["bp/", "cc/", "mf/", "all/"]

@ck.command()
@ck.option(
    '--ont-file', '-ont', default='data/go.owl',
    help='Ontology file (GO by default)')
@ck.option(
    '--dataset', '-ds',
    help="0=bp, 1=cc, 2=mf, 3=all"
)

def main(ont_file, dataset):
    dataset = int(dataset)
    if not (0 <= dataset < 4):
        raise ValueError("data folder unrecognized")

    
    train, valid, test = load_data_files(dataset)
    manager = OWLManager.createOWLOntologyManager()
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(ont_file))
    factory = manager.getOWLDataFactory()
    interacts_rel = factory.getOWLObjectProperty(
        IRI.create("http://interacts_with"))
    has_function_rel = factory.getOWLObjectProperty(
        IRI.create("http://has_function"))



    # Add GO protein annotations to the GO ontology
    
    inter_file = DATAROOT + "interactions_swissprot.tsv"
    string_uniprot_dict = DATAROOT + "uniprot_string_dict.pkl"
    with open(string_uniprot_dict, "rb") as f:
        string_uniprot_dict = pkl.load(f)
        
    with open(inter_file) as f:
        for line in f:
            inters = line.rstrip("\n").split("\t")
            p1_string, p2_string = inters[0], inters[1]  # e.g. 4932.YLR117C  and 4932.YPR101W
            
            p1_uni = string_uniprot_dict[p1_string]
            p2_uni = string_uniprot_dict[p2_string]

            for p1 in p1_uni:
                for p2 in p2_uni:

                    protein1 = factory.getOWLClass(IRI.create(f'http://{p1}'))
                    protein2 = factory.getOWLClass(IRI.create(f'http://{p2}'))
                    axiom = factory.getOWLSubClassOfAxiom(
                        protein1, factory.getOWLObjectSomeValuesFrom(
                            interacts_rel, protein2))
                    manager.addAxiom(ont, axiom)
                    axiom = factory.getOWLSubClassOfAxiom(
                        protein2, factory.getOWLObjectSomeValuesFrom(
                            interacts_rel, protein1))
                    manager.addAxiom(ont, axiom)
    
    # Add GO protein annotations to the GO ontology    
    for p_id, annots in train.items():
        protein = factory.getOWLClass(IRI.create(f'http://{p_id}'))
        for go_id in annots:
            go_id = go_id.replace(':', '_')
            go_class = factory.getOWLClass(
                IRI.create(f'http://purl.obolibrary.org/obo/{go_id}'))
            axiom = factory.getOWLSubClassOfAxiom(
                protein, factory.getOWLObjectSomeValuesFrom(
                    has_function_rel, go_class))
            manager.addAxiom(ont, axiom)
    
    # Save the files
    out_dir = DATAROOT + FOLDERS[dataset]
    new_ont_file = os.path.join(out_dir, 'ontology.owl')
    manager.saveOntology(ont, IRI.create('file:' + os.path.abspath(new_ont_file)))

    valid_ont = manager.createOntology()


    for p_id, annots in valid.items():
        protein = factory.getOWLClass(IRI.create(f'http://{p_id}'))
        for go_id in annots:
            go_id = go_id.replace(':', '_')
            go_class = factory.getOWLClass(
                IRI.create(f'http://purl.obolibrary.org/obo/{go_id}'))
            axiom = factory.getOWLSubClassOfAxiom(
                protein, factory.getOWLObjectSomeValuesFrom(
                    has_function_rel, go_class))
            manager.addAxiom(valid_ont, axiom)
       
    valid_ont_file = os.path.join(out_dir, 'valid.owl')
    manager.saveOntology(valid_ont, IRI.create('file:' + os.path.abspath(valid_ont_file)))

    test_ont = manager.createOntology()

    for p_id, annots in test.items():
        protein = factory.getOWLClass(IRI.create(f'http://{p_id}'))
        for go_id in annots:
            go_id = go_id.replace(':', '_')
            go_class = factory.getOWLClass(
                IRI.create(f'http://purl.obolibrary.org/obo/{go_id}'))
            axiom = factory.getOWLSubClassOfAxiom(
                protein, factory.getOWLObjectSomeValuesFrom(
                    has_function_rel, go_class))
            manager.addAxiom(test_ont, axiom)


    test_ont_file = os.path.join(out_dir, 'test.owl')
    manager.saveOntology(test_ont, IRI.create('file:' + os.path.abspath(test_ont_file)))


def load_data_files(dataset):

    train_data = {}
    valid_data = {}
    test_data = {}
    
    
    data_folder = DATAROOT + FOLDERS[dataset]
    with open(data_folder + "train_data_annots.tsv", "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            prot = line[0]
            annots = line[1:]

            train_data[prot] = annots

    with open(data_folder + "valid_data_annots.tsv", "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            prot = line[0]
            annots = line[1:]
            valid_data[prot] = annots

    with open(data_folder + "test_data_annots.tsv", "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            prot = line[0]
            annots = line[1:]
            test_data[prot] = annots
                
    return train_data, valid_data, test_data
            
def load_and_split_interactions(data_file, ratio=(0.9, 0.05, 0.05)):
    inter_set = set()
    with gzip.open(data_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split(' ')
            p1 = it[0]
            p2 = it[1]
            score = float(it[2])
            if score < 700:
                continue
            if (p2, p1) not in inter_set and (p1, p2) not in inter_set:
                inter_set.add((p1, p2))
    inters = np.array(list(inter_set))
    n = inters.shape[0]
    index = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(index)

    train_n = int(n * ratio[0])
    valid_n = int(n * ratio[1])
    train = inters[index[:train_n]]
    valid = inters[index[train_n: train_n + valid_n]]
    test = inters[index[train_n + valid_n:]]
    return train, valid, test
    
if __name__ == '__main__':
    main()
    shutdownJVM()
