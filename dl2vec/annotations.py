#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import Ontology, is_exp_code, is_cafa_target, FUNC_DICT

logging.basicConfig(level=logging.INFO)

DATAROOT = "data/"
FOLDERS = ["bp/", "cc/", "mf/", "all/"]
SETS = ["train_data.pkl", "valid_data.pkl", "test_data.pkl"]

@ck.command()
@ck.option(
    '--data-folder', '-df',
    help='0=bp, 1=cc, 2=mf, 3=all')

def main(data_folder):
    data_folder = int(data_folder)
    if not (0 <= data_folder < 4):
        raise ValueError("data folder unrecognized")
    else:

        for pkl_file in SETS:
            data_file = DATAROOT + FOLDERS[data_folder] + pkl_file 
            df = pd.read_pickle(data_file)
        
            out_file = DATAROOT + FOLDERS[data_folder] + pkl_file[:-4] + "_annots.tsv"
            f = open(out_file, 'w')
            for row in df.itertuples():
                prot_id = row.proteins
                #                for st_id in row.string_ids:
                f.write(prot_id)
                for go_id in row.exp_annotations:
                    f.write('\t' + go_id)
                f.write('\n')
            f.close()    
    

if __name__ == '__main__':
    main()
