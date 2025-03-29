import argparse
import numpy as np
import joblib
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
import torch
import os
import esm
import subprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PLM_ATG = joblib.load("D:/Major/AIProject/ATG/models/SVM_AADP_ESM2(400).pkl")
sorted_data = np.load("D:/Major/AIProject/ATG/data/sorted_features(aadp_esm).npz")
sorted_features = sorted_data['sorted_features']
scaler = StandardScaler()

def run_psiblast(fasta_path, pssm_path):
    command = [
        'psiblast',
        '-comp_based_stats', '1',
        '-db', 'D:/BLAST/blast-2.16.0+/db/swissprot',
        '-num_iterations', '3',
        '-inclusion_ethresh', '0.001',
        '-query',fasta_path,
        '-out_ascii_pssm',pssm_path,
        '-num_threads', '4'
    ]
    subprocess.run(command, check=True)
    print(f"\nPSSM file successfully generated: {pssm_path}")


def read_pssm_matrix(input_matrix):
    pssm = []
    with open(input_matrix) as stream:
        for line, string in enumerate(stream.readlines()):
            if line > 2:
                overall_vec = string.split()
                if len(overall_vec) == 0:
                    break

                str_vec = overall_vec[1:42]
                if len(str_vec) != 41:
                    raise ValueError("Wrong PSSM format at line {} in file {}".format(line, input_matrix))

                pssm.append(str_vec)

    pssm = np.array(pssm)
    print(f"\nLoaded PSSM matrix with shape: {pssm.shape}")
    return pssm

def average(matrix_sum, seq_len):
    matrix_array = np.divide(matrix_sum, seq_len)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1],))]
    return matrix_average

def handle_rows(pssm):
    pssm = pssm[:, 1:21]
    pssm = pssm.astype(float)

    matrix_final = np.zeros((1, 20))
    seq_cn = np.shape(pssm)[0] 

    for i in range(seq_cn):
        str_vec = pssm[i]
        if str_vec[0] == 'X':
            print(f"Skipping line in file due to unknown amino acid 'X'")
            continue

        elif str_vec[0] == 'B':
            if str_vec[3] >= str_vec[4]:
                str_vec[0] = 'N'  # 天冬氨酸 (Asp)
            else:
                str_vec[0] = 'D'  # 天冬酰胺 (Asn)
            print(f"Converting 'B' to '{str_vec[0]}' based on scores")

        elif str_vec[0] == 'Z':
            if str_vec[6] >= str_vec[7]:
                str_vec[0] = 'Q'  # 谷氨酸 (Glu)
            else:
                str_vec[0] = 'E'  # 谷氨酰胺 (Gln)
            print(f"Converting 'Z' to '{str_vec[0]}' based on scores")

        str_vec_positive = list(map(int, str_vec[0:21]))
        str_vec_positive = np.array(str_vec_positive)
        
        matrix_final[0] = list(map(sum, zip(str_vec_positive, matrix_final[0])))
        
    return matrix_final

def pre_handle_columns(pssm):
    pssm = pssm[:, 1:21]
    pssm = pssm.astype(float)

    matrix_final = np.zeros((20, 20))
    seq_cn = np.shape(pssm)[0]

    for i in range(20):
        for j in range(20):
            for k in range(seq_cn - 1):
                matrix_final[i][j] += (pssm[k][i] * pssm[k + 1][j])

    return matrix_final

def extract_aadp(pssm_path):
    pssm = read_pssm_matrix(pssm_path)
    seq_cn = float(np.shape(pssm)[0])

    aac_matrix = handle_rows(pssm)
    aac_vector = average(aac_matrix, seq_cn)

    dpc_matrix = pre_handle_columns(pssm)
    dpc_vector = average(dpc_matrix, seq_cn-1)

    aadp = np.concatenate([aac_vector[0], dpc_vector[0]])
    print(f"\nAADP feature shape: {aadp.shape}")
    return aadp


def extract_esm2(fasta_path, model_path = None):
    with open(fasta_path) as file:
        for record in SeqIO.parse(file, "fasta"):
            id = record.id
            seq = str(record.seq)
            if len(seq) > 7000:
                seq = seq[:7000]

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() 
    if(model_path is not None):
        model.load_state_dict(torch.load(model_path))
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    batch_labels, batch_strs, batch_tokens = batch_converter([(id, seq)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    sequence_representations = token_representations[0, 1:len(batch_tokens[0]) - 1].mean(0)
    
    torch.cuda.empty_cache()

    print(f"\nESM-2 embedding shape: {sequence_representations.shape}")
    return sequence_representations

def predict(fasta_path):
    pssm_path = f"{os.path.splitext(os.path.basename(fasta_path))[0]}.pssm"
    run_psiblast(fasta_path, pssm_path)

    aadp = extract_aadp(pssm_path)
    esm2 = extract_esm2(fasta_path)
    aadp_esm2 = np.concatenate((aadp, esm2))
    print(f"\nCombined AADP-ESM2 shape: {aadp_esm2.shape}")

    aadp_esm2 = aadp_esm2[sorted_features[:400]]
    aadp_esm2_scaled = scaler.fit_transform([aadp_esm2])

    print("\nStart predict...")
    prediction = PLM_ATG.predict(aadp_esm2_scaled)
    probability = PLM_ATG.predict_proba(aadp_esm2_scaled)[:, 1]

    print(f"Prediction: {prediction}")
    print(f"Prediction Probability: {probability}")
    return prediction, probability

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict ATGs from protein sequence")
    parser.add_argument('fasta_path', type=str, help='Path to the FASTA file containing a single protein sequence')
    args = parser.parse_args()

    predict(args.fasta_path)
