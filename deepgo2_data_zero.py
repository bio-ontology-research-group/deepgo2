import click as ck
import numpy as np
import pandas as pd
import os
from collections import Counter, deque
from utils import (
    Ontology, FUNC_DICT, NAMESPACES, MOLECULAR_FUNCTION, BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT, HAS_FUNCTION)
import logging
import mowl
mowl.init_jvm("64g")
import java
from mowl.owlapi import OWLAPIAdapter, Imports
from org.semanticweb.owlapi.model import IRI


logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-owl', '-go', default='data/go.owl',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_esm2.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--new-data-file', '-df', default='data/swissprot_exp_2022_04_esm2.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
def main(go_owl, go_file, data_file, new_data_file):
    owlapi = OWLAPIAdapter()
    go_owl = owlapi.owl_manager.loadOntologyFromOntologyDocument(
        java.io.File(go_owl))
    go = Ontology(go_file, with_rels=True)
    # Remove deprecated GO classes that are not in OWL
    go_cls = go_owl.getClassesInSignature(Imports.EXCLUDED)
    go_ids = set()
    for cls in go_cls:
        go_id = str(cls.toStringID())
        go_id = go_id.replace('http://purl.obolibrary.org/obo/GO_', 'GO:')
        go_ids.add(go_id)
    for go_id in go.ont:
        if go_id not in go_ids:
            del go.ont[go_id]
        
    logging.info('GO loaded')

    df = pd.read_pickle(data_file)
    proteins = set(df['proteins'].values)
    
    print("DATA FILE" ,len(df))

    new_df = pd.read_pickle(new_data_file)
    new_proteins = set(new_df['proteins'].values) - proteins
    
    print("NEW DATA FILE" ,len(new_df))
    print("NEW PROTEINS" ,len(new_proteins))

    logging.info('Processing annotations')
    test_df = new_df[new_df['proteins'].isin(new_proteins)]
    test_df = test_df.reset_index()

    for ont in ['cc', 'bp', 'mf']:
        terms_df = pd.read_pickle(f'data/{ont}/terms.pkl')
        terms_set = set(terms_df['gos'])
        index = []
        zero_index = []
        zero_terms = set()
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            zero_annots = set()
            for go_id in row.prop_annotations:
                if go.get_namespace(go_id) == NAMESPACES[ont] and \
                   go_id in go_ids:
                    if go_id in terms_set:
                        annots.add(go_id)
                    else:
                        zero_annots.add(go_id)
                    
            if len(zero_annots) > 0:
                zero_terms |= zero_annots
                zero_index.append(i)
            if len(annots) > 0:
                index.append(i)
        zero_test_df = test_df.iloc[zero_index]
        zero_terms = list(zero_terms)
        zero_df = pd.DataFrame({'gos': zero_terms})
        zero_df.to_pickle(f'data/{ont}/zero_terms.pkl')
        zero_test_df.to_pickle(f'data/{ont}/zero_test_df.pkl')
        cafa_test_df = test_df.iloc[index]
        cafa_test_df.to_pickle(f'data/{ont}/cafa_test_df.pkl')
        print(ont)
        print(len(zero_terms))
        print(len(zero_test_df), len(cafa_test_df))
        
if __name__ == '__main__':
    main()
