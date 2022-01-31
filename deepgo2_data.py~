import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, NAMESPACES, MOLECULAR_FUNCTION, BIOLOGICAL_PROCESS, CELLULAR_COMPONENT
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_zero_10.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--sim-file', '-sf', default='data/swissprot_exp.sim',
    help='Sequence similarity generated with Diamond')
def main(go_file, data_file, sim_file):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    df = pd.read_pickle(data_file)
    proteins = set(df['proteins'].values)
    
    print("DATA FILE" ,len(df))
    
    logging.info('Processing annotations')
    
    annotations = list()
    for ont in ['mf', 'bp', 'cc']:
        cnt = Counter()
        iprs = Counter()
        index = []
        for i, row in enumerate(df.itertuples()):
            ok = False
            for term in row.zero_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt[term] += 1
                    ok = True
            for ipr in row.interpros:
                iprs[ipr] += 1
            if ok:
                index.append(i)

        del cnt[FUNC_DICT[ont]] # Remove top term
        tdf = df.iloc[index]
        terms = list(cnt.keys())
        interpros = list(iprs.keys())

        print(f'Number of {ont} terms {len(terms)}')
        print(f'Number of {ont} iprs {len(iprs)}')
        print(f'Number of {ont} proteins {len(tdf)}')
    
        terms_df = pd.DataFrame({'gos': terms})
        terms_df.to_pickle(f'data/{ont}/terms_zero_10.pkl')
        continue
        iprs_df = pd.DataFrame({'interpros': interpros})
        # iprs_df.to_pickle(f'data/{ont}/interpros.pkl')

        # Split train/valid/test
        proteins = tdf['proteins']
        prot_set = set(proteins)
        prot_idx = {v:k for k, v in enumerate(proteins)}
        sim = {}
        train_prots = set()
        with open(sim_file) as f:
            for line in f:
                it = line.strip().split('\t')
                p1, p2, score = it[0], it[1], float(it[2]) / 100.0
                if p1 == p2:
                    continue
                if score < 0.5:
                    continue
                if p1 not in prot_set or p2 not in prot_set:
                    continue
                if p1 not in sim:
                    sim[p1] = []
                if p2 not in sim:
                    sim[p2] = []
                sim[p1].append(p2)
                sim[p2].append(p1)

        used = set()
        def dfs(prots, prot):
            used.add(prot)
            if prot in sim:
                for p in sim[prot]:
                    if p not in used:
                        dfs(prots, p)
            prots.append(prot)

        groups = []
        for p in proteins:
            group = []
            if p not in used:
                dfs(group, p)
                groups.append(group)
        print(len(proteins), len(groups))
        index = np.arange(len(groups))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        train_n = int(len(groups) * 0.9)
        valid_n = int(train_n * 0.9)
    
        train_index = []
        valid_index = []
        test_index = []
        for idx in index[:valid_n]:
            for prot in groups[idx]:
                train_index.append(prot_idx[prot])
        for idx in index[valid_n:train_n]:
            for prot in groups[idx]:
                valid_index.append(prot_idx[prot])
        for idx in index[train_n:]:
            for prot in groups[idx]:
                test_index.append(prot_idx[prot])
                
        train_index = np.array(train_index)
        valid_index = np.array(valid_index)
        test_index = np.array(test_index)

        train_df = tdf.iloc[train_index]
        train_df.to_pickle(f'data/{ont}/train_data.pkl')
        valid_df = tdf.iloc[valid_index]
        valid_df.to_pickle(f'data/{ont}/valid_data.pkl')
        test_df = tdf.iloc[test_index]
        test_df.to_pickle(f'data/{ont}/test_data.pkl')

        print(f'Train/Valid/Test proteins for {ont} {len(train_df)}/{len(valid_df)}/{len(test_df)}')

    
if __name__ == '__main__':
    main()
