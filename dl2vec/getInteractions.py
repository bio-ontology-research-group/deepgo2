#!/usr/bin/env python3

##################################################################
## For the given list of proteins print out only the interactions
## between these protein which have medium or higher confidence
## experimental score
##
## Requires requests module:
## type "python -m pip install requests" in command line (win)
## or terminal (mac/linux) to install the module
##################################################################

import click as ck
import requests ## python -m pip install requests
import pickle as pkl
import logging
from itertools import chain




@ck.command()
@ck.option(
    '--verbose', '-v', default=False,
    help="If true, information output will be printed during execution"
)
def main(verbose):
    processDataSet(verbose)


DATAROOT = 'data/'
SWISSPROT = 'swissprot_exp.pkl'
OUTFILE = 'network.csv'


def processDataSet(verbose = False):
    
                                                
    if verbose:
        logging.basicConfig(level = logging.INFO)

    pickle_filename = DATAROOT + SWISSPROT
    logging.info("Working on file %s", pickle_filename)
        
    with open(pickle_filename, "rb") as f:
        df = pkl.load(f)

    prots = df["string_ids"].tolist()
    prots = set(chain(*prots))

    with open("data/interactions.txt", "r") as f:
        outfile = DATAROOT + OUTFILE
        logging.info("Saving network in %s", outfile)
        with open(outfile, "w") as out:
            for line in f:
                line = line.rstrip("\n").split(" ")
                if len(line) < 2:
                    continue
                prot1, prot2 = line[0], line[1]
                if prot1 in prots and prot2 in prots:
                    out.write(prot1 + "," + prot2 + "\n")

                
def get_network2(proteins, threshold = 0.7):

    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"

    ##
    ## Construct URL
    ##

    request_url = "/".join([string_api_url, output_format, method])

    params = {
        "identifiers" : "%0d".join(proteins), # your protein
        "caller_identity" : "www.awesome_app.org", # your app name
        "add_color_nodes": 10,
        "add_white_nodes": 10
    }

    ##
    ## Call STRING
    ##

    response = requests.post(request_url, data=params)
    edges = list()

    if response.ok:
        for line in response.text.strip().split("\n"):

            l = line.strip().split("\t")
            if len(l) == 13:
                p1, p2, prob = l[2], l[3], l[10]
        
                ## filter the interaction according to experimental score
                experimental_score = float(prob)
                if experimental_score >= threshold:
                    edges.append((p1, p2))
            else:
                print(l)
    else:
        print(response.text)    
    return edges


if __name__=="__main__":

    main()
