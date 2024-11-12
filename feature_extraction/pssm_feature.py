import os
import numpy as np
import pandas as pd
from Bio import SeqIO

all_encoders = ['aac', 'dpc']

# 从FASTA文件中读取蛋白质名称
def get_protein_names(fasta_file):
    protein_name_mapping = {}
    for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        protein_name_mapping[idx] = record.id  # 将编号映射到完整的蛋白质名称
    return protein_name_mapping

#读取PSSM文件并将其内容转换为NumPy数组
def read_pssm_matrix(input_matrix):
    PSSM = []
    with open(input_matrix) as stream:
        for line, string in enumerate(stream.readlines()):
            if line > 2:
                overall_vec = string.split()
                if len(overall_vec) == 0:
                    break

                str_vec = overall_vec[1:42]
                if len(str_vec) != 41:
                    raise ValueError("Wrong PSSM format at line {} in file {}".format(line, input_matrix))

                PSSM.append(str_vec)

    PSSM = np.array(PSSM)
    return PSSM

# 将生成的矩阵取平均值，并转换成一维向量
def average(matrix_sum, seq_len):
    matrix_array = np.divide(matrix_sum, seq_len)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1],))]
    return matrix_average

# 处理行，计算每种氨基酸的得分之和
def handle_rows(pssm, switch, count):
    """
    if SWITCH=0, we filter no element.
    if SWITCH=1, we filter all the negative elements.
    if SWITCH=2, we filter all the negative and positive elements greater than expected.
    if COUNT=20, we generate a 20-dimension vector.
    if COUNT=400, we generate a 400-dimension vector.
    """

    Amino_vec = "ARNDCQEGHILKMFPSTWYV" #20种氨基酸，用于映射氨基酸的索引位置

    matrix_final = np.zeros((int(count / 20), 20))
    seq_cn = 0 #用于记录PSSM矩阵的行数（序列长度）

    PSSM_shape = np.shape(pssm)
    for i in range(PSSM_shape[0]):
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

        seq_cn += 1
        str_vec_positive = list(map(int, str_vec[1:21]))
        str_vec_positive = np.array(str_vec_positive)
        if switch == 1:
            str_vec_positive[str_vec_positive < 0] = 0
        elif switch == 2:
            str_vec_positive[str_vec_positive < 0] = 0
            str_vec_positive[str_vec_positive > 7] = 0
        if count == 20:
            matrix_final[0] = list(map(sum, zip(str_vec_positive, matrix_final[0])))
        elif count == 400:
            try:
                amino_index = Amino_vec.index(str_vec[0])
            except ValueError as e:
                print(f"Error: {e}. Amino acid {str_vec[0]} not found in Amino_vec. Line content: {str_vec}")
                raise
            matrix_final[amino_index] = list(map(sum, zip(str_vec_positive, matrix_final[amino_index])))
        else:
            raise ValueError("Invalid count value: {}. It should be either 20 or 400.".format(count))
        
    return matrix_final,seq_cn

# 处理列，计算氨基酸对的关系
def pre_handle_columns(pssm, part):
    """
    if part=0, we calculate the left part of PSSM.
    if part=1, we calculate the right part of PSSM.

    """
    # 选择PSSM矩阵的部分
    if part == 0:
        pssm = pssm[:, 1:21]
    elif part == 1:
        pssm = pssm[:, 21:]
    pssm = pssm.astype(float)

    matrix_final = np.zeros((20, 20))
    seq_cn = np.shape(pssm)[0]

    for i in range(20):
        for j in range(20):
            for k in range(seq_cn - 1):
                matrix_final[i][j] += (pssm[k][i] * pssm[k + 1][j])

    return matrix_final

# AAC PSSM feature encoder
def aac(pssm):
    SWITCH = 0
    COUNT = 20
    aac_matrix, seq_cn = handle_rows(pssm, SWITCH, COUNT)
    aac_matrix = np.array(aac_matrix)
    aac_vector = average(aac_matrix, seq_cn)
    return aac_vector[0]

# DPC PSSM feature encoder
def dpc(pssm):
    PART = 0
    matrix_final = pre_handle_columns(pssm, PART)
    seq_cn = float(np.shape(pssm)[0])
    dpc_vector = average(matrix_final, seq_cn-1)
    return dpc_vector[0]


# 从指定目录中的PSSM文件提取特定类型的特征，并将结果保存为CSV文件
def get_feature(pssm_dir, algo_type, output_file, protein_name_mapping):
    print("Processing directory: {}".format(pssm_dir))

    # 获取目录中所有文件
    pssm_files = [os.path.join(pssm_dir, pf.name) for pf in os.scandir(pssm_dir) if not pf.is_dir()]
    # 读取并解析所有 PSSM 文件
    pssm_mat = list(map(read_pssm_matrix, pssm_files))
    # 提取特征
    features = np.array(list(map(eval(algo_type), pssm_mat)))
    print("Extracted features using {}: {}".format(algo_type, features.shape))

    protein_names = [protein_name_mapping[int(os.path.splitext(pf.name)[0])] for pf in os.scandir(pssm_dir) if not pf.is_dir()]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file,'w') as f:
        for pn, feats in zip(protein_names, features):
            f.write(pn + ",")
            f.write(",".join(list(map(str, feats))))
            f.write("\n")
    print("Saved features to file: {}".format(output_file))

# 从指定目录中的PSSM文件提取所有类型的特征，并将结果保存为多个CSV文件
def process_pssm_folders(base_dir):
    datasets = {
        "Negative_test/pssm_profile_uniref50": "ne_test",
        "Negative_training/pssm_profile_uniref50": "ne_train",
        "Positive_test/pssm_profile_uniref50": "po_test",
        "Positive_training/pssm_profile_uniref50": "po_train"
    }
    
    for dataset, suffix in datasets.items():
        fasta_name = dataset.split("/")[0] + ".fasta"
        fasta_file = os.path.join("D:/Major/AIProject/ATG/data/fastadata",fasta_name)

        protein_name_mapping = get_protein_names(fasta_file)

        for encoder in all_encoders:
            output_file = os.path.join(base_dir,f"{encoder}_{suffix}.csv")
            input_dir = os.path.join(base_dir, dataset)
            print("\n\nProcessing {} dataset, using {} encoder".format(dataset, encoder))
            get_feature(input_dir, encoder, output_file, protein_name_mapping)

def merge_features(base_dir):
    datasets = {
        "Negative_test": "ne_test",
        "Negative_training": "ne_train",
        "Positive_test": "po_test",
        "Positive_training": "po_train"
    }
    
    for dataset, suffix in datasets.items():
        # 定义输入文件路径
        aac_file = os.path.join(base_dir, f"aac_{suffix}.csv")
        dpc_file = os.path.join(base_dir, f"dpc_{suffix}.csv")
        
        # 读取CSV文件
        aac_df = pd.read_csv(aac_file, header=None)
        dpc_df = pd.read_csv(dpc_file, header=None)

        # 检查是否有相同的Uniprot_id列
        assert all(aac_df.iloc[:, 0] == dpc_df.iloc[:, 0]), "Mismatch in Uniprot_id between AAC and DPC"
        
        # 删除重复的Uniprot_id和label列
        aac_features = aac_df.iloc[:, 1:]  # 除第一列外的所有列
        dpc_features = dpc_df.iloc[:, 1:]  # 除第一列外的所有列
        
        # 合并特征
        aadp_features = pd.concat([aac_df.iloc[:, 0], aac_features, dpc_features], axis=1)
        
        # 保存为CSV文件
        output_file = os.path.join(base_dir, f"aadp_{suffix}.csv")
        aadp_features.to_csv(output_file, header=False, index=False)
        num_columns = aadp_features.shape[1]
        print(f"Saved aadp features to file: {output_file}, {num_columns} columns.")


if __name__ == "__main__":
    base_dir = "D:/Major/AIProject/ATG/data/pssmdata"
    process_pssm_folders(base_dir)
    merge_features(base_dir)