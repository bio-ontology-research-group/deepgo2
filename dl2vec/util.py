import gzip
import pickle as pkl
from itertools import chain

def higherThan700():

    filepath = 'data/protein.links.v11.5.txt.gz'

    fout = open('data/interactions.txt', "w")

    with gzip.open(filepath, "rt") as f:
        for line in f:
            value = line.rstrip("\n").split(" ")[2]
            if value == "combined_score":
                continue
            if int(value) >= 700:
                fout.write(line+'\n')

def inters_swissprot():
    inters_file = "data/interactions.txt"
    swissprot = "data/swissprot_exp.pkl"
    out = "data/interactions_swissprot.tsv"

    swiss_file = open(swissprot, "rb")
    df_swiss = pkl.load(swiss_file)
    prots_swiss = df_swiss["string_ids"]
    prots_swiss = set(chain(*prots_swiss))

    with open(out, "w") as fout:
        with open(inters_file, "r") as fin:
            for line in fin:
                
                line = line.rstrip("\n").split(" ")

                if len(line) == 3:
                    p1, p2 = line[0], line[1]

                    if p1 in prots_swiss and p2 in prots_swiss:
                        fout.write(p1 + "\t" + p2 + "\n")
    
def uni_string_dict():
    out = "data/uniprot_string_dict.pkl"
    infile = open("data/swissprot_exp.pkl", "rb")

    df = pkl.load(infile)

    out_dict = {}
    for row in df.itertuples():
        prot = row.proteins
        ids = row.string_ids
        for id in ids:
            if not id in out_dict:
                out_dict[id] = list()
            out_dict[id].append(prot)

    with open(out, "wb") as f:
        pkl.dump(out_dict, f)



                
def get_swissprot_prots():
    with open("data/swissprot_exp.pkl", "rb") as f:
        df = pkl.load(f)

        prots = df["string_ids"]
        prots = set(chain(*prots))
        print(f"Number of proteins: {len(prots)}")

    with open("data/prots.pkl", "wb") as f:
        pkl.dump(prots, f)

                
def removeEmptyLines():
    filepath = 'data/interactions.txt'
    fout = open('data/interactions_good.txt', "w")

    with open(filepath, "r") as f:
        for line in f:
            if line.rstrip("\n") == "":
                continue
            fout.write(line)
 


def findAlias(protein):
    filepath = 'data/string/protein.aliases.v11.5.txt.gz'

    with gzip.open(filepath, "rt") as f:
        for line in f:
            if protein in line:
                print(line)
