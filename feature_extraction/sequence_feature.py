import os
import pandas as pd
from Bio import SeqIO
from itertools import product

amino_acids = 'ARNDCQEGHILKMFPSTWYV'

def get_aac(sequence):
    aac = []
    sequence = sequence.replace('X', '')
    seq_len = len(sequence)

    for aa in amino_acids:
        aac.append(sequence.count(aa) / seq_len)
    
    return aac

def get_dpc(sequence):
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    dpc = []
    sequence = sequence.replace('X', '')
    seq_len = len(sequence) - 1
    
    # 初始化双肽的字典
    dp_count = {dp: 0 for dp in dipeptides}

    # 计算每个双肽的出现次数
    for i in range(seq_len):
        dp = sequence[i:i+2]
        if dp in dp_count:
            dp_count[dp] += 1
    # 标准化     
    for dp in dipeptides:
        dpc.append(dp_count[dp] / seq_len)
        
    return dpc

def read_fasta(file_path):
    sequences = []
    names = []
    for record in SeqIO.parse(file_path, "fasta"):
        names.append(record.id)
        sequences.append(str(record.seq))
    return  names, sequences

def save_features_to_csv(file_path, protein_names, features):
    df = pd.DataFrame(features)
    df.insert(0, "Protein_ID", protein_names)
    df.to_csv(file_path, header=False, index=False)

    num_rows, num_columns = df.shape
    print(f"Saved features to file: {file_path}, {num_rows} rows, {num_columns} columns.")

def process_fasta_files(base_dir):
    datasets = {
        # "Negative_test.fasta": "ne_test",
        "Negative_training.fasta": "ne_train",
        # "Positive_test.fasta": "po_test",
        # "Positive_training.fasta": "po_train"
    }

    for fasta_file, suffix in datasets.items():
        inputdir = os.path.join(base_dir, fasta_file)
        names, sequences = read_fasta(inputdir)
        
        aac_features = [get_aac(seq) for seq in sequences]
        dpc_features = [get_dpc(seq) for seq in sequences]
        aadp_features = [aac + dpc for aac, dpc in zip(aac_features, dpc_features)]
        
        aac_output_file = os.path.join(base_dir, f"aac_{suffix}.csv")
        save_features_to_csv(aac_output_file, names, aac_features)

        dpc_output_file = os.path.join(base_dir, f"dpc_{suffix}.csv")
        save_features_to_csv(dpc_output_file, names, dpc_features)
        
        aadp_output_file = os.path.join(base_dir, f"aadp_{suffix}.csv")
        save_features_to_csv(aadp_output_file, names, aadp_features)

if __name__ == "__main__":
    base_dir = "D:/Major/AIProject/ATG/data/fastadata"
    process_fasta_files(base_dir)

# Saved features to file: D:/Major/AIProject/ATG/data/fastadata\aac_ne_train.csv, 357 rows, 21 columns.
# Saved features to file: D:/Major/AIProject/ATG/data/fastadata\dpc_ne_train.csv, 357 rows, 401 columns.
# Saved features to file: D:/Major/AIProject/ATG/data/fastadata\aadp_ne_train.csv, 357 rows, 421 columns.