import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, NAMESPACES, read_fasta, CAFA_TARGETS
import logging
import os
import gzip

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data-netgo/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data-netgo/uniprot.tab.gz',
    help='Uniprot KB, generated with uni2pandas.py')
def main(go_file, data_file):
    go = Ontology(go_file, with_rels=True)
    
    interpros = {}
    sequences = {}
    orgs = {}
    reviewed = set()
    with gzip.open(data_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            if len(it) < 2:
                continue
            p_id = it[0]
            org_id = it[4]
            orgs[p_id] = org_id
            sequences[p_id] = it[1]
            if it[3] == 'reviewed':
                reviewed.add(p_id)
            if len(it) > 2:
                iprs = [ipr for ipr in it[2].split(';') if ipr]
            else:
                iprs = []
            interpros[p_id] = iprs

    print(len(reviewed))
    print("Loading NetGO2 data")
    
    def load_netgo_proteins(filename, is_test=False):
        proteins = {}
        with open(f'data-netgo/{filename}') as f:
            for line in f:
                it = line.strip().split('\t')
                p_id, g_id = it[0], it[1]
                if is_test:
                    if p_id not in orgs or orgs[p_id] not in CAFA_TARGETS:
                        continue
                if p_id not in proteins:
                    proteins[p_id] = []
                proteins[p_id].append(g_id)

        if is_test:
            # Remove proteins with only binding annotation
            prots = list(proteins.keys())
            for p_id in prots:
                if len(proteins[p_id]) == 1 and proteins[p_id][0] == 'GO:0005515':
                    del proteins[p_id]

        seqs = []
        prots = []
        exp_annotations = []
        prop_annotations = []
        iprs = []
        for p_id, annots in proteins.items():
            if p_id not in sequences:
                continue
            prots.append(p_id)
            exp_annotations.append(annots)
            seqs.append(sequences[p_id])
            annot_set = set()
            for go_id in annots:
                annot_set |= go.get_anchestors(go_id)
            annots = list(annot_set)
            prop_annotations.append(annots)
            if p_id in interpros:
                iprs.append(interpros[p_id])
            else:
                iprs.append(set())
        df = pd.DataFrame({
            'proteins': prots,
            'sequences': seqs,
            'exp_annotations': exp_annotations,
            'prop_annotations': prop_annotations,
            'interpros': iprs
        })
        return df

    train_df = load_netgo_proteins('train.txt')
    valid_df = load_netgo_proteins('valid.txt')
    test_df = load_netgo_proteins('test.txt', is_test=True)
    
        
    print('Processing train and valid annotations')
    
    annotations = list()
    for ont in ['mf', 'bp', 'cc']:
        cnt = Counter()
        iprs = Counter()
        train_index = []
        valid_index = []
        test_index = []
        known_prots = set()
        for i, row in enumerate(train_df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt[term] += 1
                    ok = True
            for ipr in row.interpros:
                iprs[ipr] += 1
            if ok:
                train_index.append(i)
            known_prots.add(row.proteins)
                
        for i, row in enumerate(valid_df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt[term] += 1
                    ok = True
            for ipr in row.interpros:
                iprs[ipr] += 1
            if ok and row.proteins not in known_prots:
                valid_index.append(i)
            known_prots.add(row.proteins)

        for i, row in enumerate(test_df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    ok = True
                if len(row.interpros) == 0:
                    ok = False
            if ok and row.proteins not in known_prots:
                test_index.append(i)

            
        del cnt[FUNC_DICT[ont]] # Remove top term
        terms = list(cnt.keys())
        interpros = list(iprs.keys())

        print(f'Number of {ont} terms {len(terms)}')
        print(f'Number of {ont} iprs {len(iprs)}')
        
        terms_df = pd.DataFrame({'gos': terms})
        terms_df.to_pickle(f'data-netgo/{ont}/terms.pkl')
        iprs_df = pd.DataFrame({'interpros': interpros})
        iprs_df.to_pickle(f'data-netgo/{ont}/interpros.pkl')

        tr_df = train_df.iloc[train_index]
        tr_df.to_pickle(f'data-netgo/{ont}/train_data.pkl')
        vl_df = valid_df.iloc[valid_index]
        vl_df.to_pickle(f'data-netgo/{ont}/valid_data.pkl')
        ts_df = test_df.iloc[test_index]
        ts_df.to_pickle(f'data-netgo/{ont}/test_data.pkl')

        with open(f'data-netgo/{ont}/{ont}_test.fasta', 'w') as f:
            for row in ts_df.itertuples():
                f.write(f'>{row.proteins}\n{row.sequences}\n\n')
        with open(f'data-netgo/{ont}/{ont}_test_go.txt', 'w') as f:
            for row in ts_df.itertuples():
                for go_id in row.prop_annotations:
                    if go.get_namespace(go_id) == NAMESPACES[ont]:
                        f.write(f'{row.proteins}\t{go_id}\t{ont}\n')
                
        
        print(f'Train/Valid proteins for {ont} {len(tr_df)}/{len(vl_df)}')
        print(f'Test proteins for {ont} {len(ts_df)}')


if __name__ == '__main__':
    main()
