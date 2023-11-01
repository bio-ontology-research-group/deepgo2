#!/usr/bin/env python
import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
import gzip

from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='Pandas dataframe with protein sequences')
@ck.option(
    '--out-file', '-of', default='data/swissprot_exp.fa',
    help='Fasta file')
def main(data_file, out_file):
    # Load interpro data
    df = pd.read_pickle(data_file)
    print(len(df)) 
    with open(out_file, 'w') as f:
        for row in df.itertuples():
            prot_id = row.proteins
            f.write('>' + prot_id + '\n')
            f.write(row.sequences + '\n')
    

if __name__ == '__main__':
    main()
