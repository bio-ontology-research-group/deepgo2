import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
import os
from collections import Counter, deque
from deepgo.utils import (
    Ontology, FUNC_DICT, NAMESPACES, MOLECULAR_FUNCTION, BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT, HAS_FUNCTION)


logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/nextprot_mf.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
def main(go_file, data_file):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')
    df = pd.read_pickle(data_file)
    proteins = set(df['proteins'].values)
    
    print("DATA FILE" ,len(df))
    
    logging.info('Processing annotations')
    
    annotations = list()
    for ont in ['mf', 'bp',]:
        train_df = pd.read_pickle(f'data/{ont}/train_data.pkl')
        valid_df = pd.read_pickle(f'data/{ont}/valid_data.pkl')
        train_df = pd.concat([train_df, valid_df])
        train_proteins = set(train_df['proteins'])
        cnt = Counter()
        index = []
        for i, row in enumerate(df.itertuples()):
            if row.proteins in train_proteins:
                continue
            ok = False
            for term in row.prop_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    ok = True
                    break
            if ok:
                index.append(i)

        ont_df = df.iloc[index]
        print(ont_df)
        ont_df.to_pickle(f'data/{ont}/nextprot_data.pkl')
                
if __name__ == '__main__':
    main()
