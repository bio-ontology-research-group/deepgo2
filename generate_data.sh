!#/bin/sh
data_root=data

echo "Parsing UniProt data and creating the DataFrame"
python gendata/uni2pandas.py -sf $data_root/uniprot_sprot.dat.gz -o $data_root/swissprot_exp.pkl

echo "Adding interactions data to the DataFrame"
python gendata/ppi_data.py

echo "Creating FASTA file for diamond database"
python gendata/pkl2fasta.py -df $data_root/swissprot_exp.pkl -o $data_root/swissprot_exp.fa

echo "Creating Diamond database and compute similarities"
diamond makedb --in $data_root/swissprot_exp.fa --db $data_root/swissprot_exp.dmnd
diamond blastp --very-sensitive -d $data_root/swissprot_exp.dmnd -q $data_root/$ont/train_data.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/swissprot_exp.sim

echo "Splitting train/valid/test"
python gendata/deepgo2_data.py
