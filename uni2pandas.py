#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import Ontology, is_exp_code, is_cafa_target, FUNC_DICT

logging.basicConfig(level=logging.INFO)

ORGS = set(['HUMAN', 'MOUSE', ])

@ck.command()
@ck.option(
    '--swissprot-file', '-sf', default='data/uniprot_sprot_2021_04.dat.gz',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
@ck.option(
    '--out-file', '-o', default='data/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')
def main(swissprot_file, out_file):
    go = Ontology('data/go.obo', with_rels=True)
    proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros = load_data(swissprot_file)
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'genes': genes,
        'sequences': sequences,
        'annotations': annotations,
        'string_ids': string_ids,
        'orgs': orgs,
        'interpros': interpros
    })

    logging.info('Filtering proteins with experimental annotations')
    index = []
    annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
        # Ignore proteins without experimental annotations
        # if len(annots) == 0:
        #     continue
        # index.append(i)
        annotations.append(annots)
    # df = df.iloc[index]
    # df = df.reset_index()
    df['exp_annotations'] = annotations

    prop_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            annot_set |= go.get_anchestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations

    cafa_target = []
    for i, row in enumerate(df.itertuples()):
        if is_cafa_target(row.orgs):
            cafa_target.append(True)
        else:
            cafa_target.append(False)
    df['cafa_target'] = cafa_target
    
    df.to_pickle(out_file)
    logging.info('Successfully saved %d proteins' % (len(df),) )

    
    
def load_data(swissprot_file):
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    string_ids = list()
    orgs = list()
    genes = list()
    interpros = list()
    with gzip.open(swissprot_file, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        strs = list()
        iprs = list()
        gene_id = ''
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    string_ids.append(strs)
                    orgs.append(org)
                    genes.append(gene_id)
                    interpros.append(iprs)
                prot_id = items[1]
                annots = list()
                strs = list()
                iprs = list()
                seq = ''
                gene_id = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                elif items[0] == 'STRING':
                    str_id = items[1]
                    strs.append(str_id)
                elif items[0] == 'GeneID':
                    gene_id = items[1]
                elif items[0] == 'InterPro':
                    ipr_id = items[1]
                    iprs.append(ipr_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        string_ids.append(strs)
        orgs.append(org)
        genes.append(gene_id)
        interpros.append(iprs)
    return proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros


if __name__ == '__main__':
    main()
