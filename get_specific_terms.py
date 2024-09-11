#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from deepgo.utils import Ontology, NAMESPACES
from collections import defaultdict

@ck.command()
@ck.option('--data-root', '-d', default='data')
@ck.option('--in-file', '-i', help='Input file', required=True)
@ck.option('--out-file', '-o', help='Output file', required=True)
def main(data_root, in_file, out_file):
    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    # Dictionary to hold the data for each identifier
    data = defaultdict(list)

    # Read the input file
    with open(in_file, 'r') as f:
        for line in f:
            identifier, go_id, score = line.strip().split()
            data[identifier].append((go_id, float(score)))

    # Open the output file
    with open(out_file, 'w') as w:
        for identifier, go_terms in data.items():
            go_set = set([x[0] for x in go_terms])
            scores = {x[0]: x[1] for x in go_terms}

            # Filter out ancestors
            for go_id in list(go_set):
                ancestors = go.get_ancestors(go_id)
                ancestors.discard(go_id)
                go_set -= ancestors

            # Write to the output file
            for go_id in go_set:
                w.write(f"{identifier}\t{go_id}\t{scores[go_id]:.3f}\n")

if __name__ == '__main__':
    main()
