from transformers import T5EncoderModel, T5Tokenizer
import torch
from Bio import SeqIO
import gc #垃圾回收模块
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化 T5 模型和分词器
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = model.to(device)
    model = model.eval()
    return model, tokenizer

'''
获取蛋白质序列的嵌入
model: 预训练的 ProtT5 模型
tokenizer: ProtT5 模型的分词器
seqs: 序列字典，包含蛋白质序列数据
per_residue: 是否提取每个残基的嵌入特征（布尔值）
per_protein: 是否提取每个蛋白质的嵌入特征（布尔值）
sec_struct: 是否预测二级结构
max_residues: 每个批次中允许的最大残基数量
max_seq_len: 每个序列的最大长度
max_batch: 每个批次中允许的最大序列数量
'''
def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=4600, max_batch=1):
    res_names = []
    results = {"residue_embs": dict(),"protein_embs": dict(),"sec_structs": dict()}

    # 根据长度对序列进行排序（减少不必要的填充，提高嵌入速度）
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    
    batch = list()
    for seq_idx, (id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)

        # 如果序列长度超过了 max_seq_len，则进行截断
        if seq_len > max_seq_len:
            print(f"Truncating sequence {id} from length {seq_len} to {max_seq_len}")
            seq = seq[:max_seq_len]
            seq_len = max_seq_len

        seq = ' '.join(list(seq))
        batch.append((id, seq, seq_len))

        # 计算当前批次中的残基数，并避免处理残基数量超过 max_residues 的批次
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            # 清空批次列表
            ids, seqs, seq_lens = zip(*batch)
            res_names.append(ids)
            batch = list()

            # 使用分词器对批次中的序列进行编码,并填充到最长序列的长度
            # 返回一个包含 input_ids 和 attention_mask 的字典
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                # 在不计算梯度的上下文中执行，减少内存使用和计算时间
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(id, seq_len))
                continue

            # 存储嵌入特征
            for batch_idx, identifier in enumerate(ids):  # 遍历当前批次中的序列 ID
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:  # store per-residue embeddings (L x 1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()
            
            # 清理缓存
            del embedding_repr, token_encoding
            torch.cuda.empty_cache()
            gc.collect()

    return results, res_names

# 从FASTA文件中读取蛋白质序列
def get_seq(file_path):
    seqs = [] # 用于存储序列数据的列表
    names = []  # 用于存储序列名称的列表

    with open(file_path) as file:
        for record in SeqIO.parse(file, "fasta"):
            names.append(record.id)
            seqs.append(str(record.seq)) # 将序列数据添加到 seqs 列表中
        # 将 names 列表和 seqs 列表组合成一个字典
        seq_dict = dict(zip(names, seqs))
        print(f"{file_path}: {len(seq_dict)} sequences")
    return seq_dict

def process_and_save(model, tokenizer, fasta_path, save_path):
    seq_dict = get_seq(fasta_path)
    results, _ = get_embeddings(model, tokenizer, seq_dict, per_residue=0, per_protein=1, sec_struct=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **results["protein_embs"])
    print(f"Saved embeddings to {save_path}")

if __name__ == '__main__':
    model, tokenizer = get_T5_model()
    gc.collect() # 清理内存，收集垃圾

    file_paths = {
        "D:/Major/AIProject/ATG/data/pretrain/Test/t5_ne.npz":
          "D:/Major/AIProject/ATG/data/fastadata/Negative_test.fasta",

        "D:/Major/AIProject/ATG/data/pretrain/Test/t5_po.npz":
          "D:/Major/AIProject/ATG/data/fastadata/Positive_test.fasta",

        "D:/Major/AIProject/ATG/data/pretrain/Train/t5_ne.npz":
          "D:/Major/AIProject/ATG/data/fastadata/Negative_training.fasta",

        "D:/Major/AIProject/ATG/data/pretrain/Train/t5_po.npz":
         "D:/Major/AIProject/ATG/data/fastadata/Positive_training.fasta"
    }
    for save_path, fasta_path in file_paths.items():
        process_and_save(model, tokenizer, fasta_path, save_path)


# You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
# /usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
#   warnings.warn(
# /content/drive/MyDrive/autophagy_protein/data/fastadata/Negative_test.fasta: 100 sequences
# Truncating sequence tr|A8J1H4|A8J1H4_CHLRE from length 7560 to 4600
# Truncating sequence tr|C7PT76|C7PT76_CHIPD from length 7122 to 4600
# Truncating sequence tr|O11993|O11993_BVDV from length 4983 to 4600
# Truncating sequence tr|C7H0E5|C7H0E5_9FIRM from length 4743 to 4600
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Test/t5_ne.npz
# /content/drive/MyDrive/autophagy_protein/data/fastadata/Positive_test.fasta: 100 sequences
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Test/t5_po.npz
# /content/drive/MyDrive/autophagy_protein/data/fastadata/Negative_training.fasta: 357 sequences
# Truncating sequence tr|Q3AR72|Q3AR72_CHLCH from length 16311 to 4600
# Truncating sequence tr|Q8IB94|Q8IB94_PLAF7 from length 8591 to 4600
# Truncating sequence tr|Q8IM09|Q8IM09_PLAF7 from length 7182 to 4600
# Truncating sequence tr|A9V949|A9V949_MONBE from length 7041 to 4600
# Truncating sequence tr|A4I2N6|A4I2N6_LEIIN from length 6168 to 4600
# Truncating sequence tr|A8JA83|A8JA83_CHLRE from length 5684 to 4600
# Truncating sequence tr|A5L6I6|A5L6I6_9GAMM from length 5428 to 4600
# Truncating sequence tr|D3BRE2|D3BRE2_POLPA from length 5187 to 4600
# Truncating sequence tr|A0C7B3|A0C7B3_PARTE from length 5133 to 4600
# Truncating sequence tr|D0MYQ3|D0MYQ3_PHYIN from length 5129 to 4600
# Truncating sequence tr|C7PYI8|C7PYI8_CATAD from length 5128 to 4600
# Truncating sequence tr|Q4Q3Q7|Q4Q3Q7_LEIMA from length 5095 to 4600
# Truncating sequence tr|D2VP19|D2VP19_NAEGR from length 5057 to 4600
# Truncating sequence tr|A2BH96|A2BH96_HUMAN from length 4621 to 4600
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Train/t5_ne.npz
# /content/drive/MyDrive/autophagy_protein/data/fastadata/Positive_training.fasta: 393 sequences
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Train/t5_po.npz