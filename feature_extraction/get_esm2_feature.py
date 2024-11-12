import numpy as np
import esm
import torch
from Bio import SeqIO
from tqdm import tqdm
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def esm_embeddings(seqs_list, model_path = None, batch_size=1):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 加载预训练模型
    if(model_path is not None):
        model.load_state_dict(torch.load(model_path))
    batch_converter = alphabet.get_batch_converter() # 获取批处理转换器
    model.eval()
    model.to(device)

    sequence_representations = []

    # 分批处理序列
    for i in tqdm(range(0, len(seqs_list), batch_size), desc="Processing batches"):
        batch = seqs_list[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # 计算每个序列的长度
        batch_tokens = batch_tokens.to(device)

        # 提取嵌入特征
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # 处理每个序列的嵌入特征
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(
                (
                    batch[i][0],
                    token_representations[i, 1:tokens_len - 1].mean(0),
                )
            )
        # 清理缓存
        torch.cuda.empty_cache()

    return sequence_representations #包含序列ID及其对应嵌入的列表

def get_seqs_list(file_path):
    seqs_list = []
    with open(file_path) as file:
        for record in SeqIO.parse(file, "fasta"):
            id = record.id
            seq = str(record.seq)
            if len(seq) > 7000:
                seq = seq[:7000]  # 截断过长的序列
            seqs_list.append((id, seq))
    print(f"{file_path}: {len(seqs_list)} sequences loaded")
    return seqs_list

def process_and_save(fasta_path, save_path):
    seqs_list = get_seqs_list(fasta_path)

    # 对序列按长度进行升序
    seqs_list = sorted(seqs_list, key=lambda item: len(item[1]))

    embeddings = esm_embeddings(seqs_list)
    embeddings_dict = {seq_id: embedding.cpu().numpy() for seq_id, embedding in embeddings}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **embeddings_dict)
    print(f"Saved embeddings to {save_path}")

    torch.cuda.empty_cache()
    if embeddings:
        print(f"First embedding shape: {embeddings[0][1].shape}")
        print(f"Total sequences processed: {len(embeddings)}\n")

if __name__ == "__main__":
    file_paths = {
        "/root/ATG/data/pretrain/Test/esm2_ne.npz":
          "/root/ATG/data/fastadata/Negative_test.fasta",

        "/root/ATG/data/pretrain/Test/esm2_po.npz":
          "/root/ATG/data/fastadata/Positive_test.fasta",

        "/root/ATG/data/pretrain/Train/esm2_ne.npz":
          "/root/ATG/data/fastadata/Negative_training.fasta",

        "/root/ATG/data/pretrain/Train/esm2_po.npz":
         "/root/ATG/data/fastadata/Positive_training.fasta"
    }

    for save_path, fasta_path in file_paths.items():
        process_and_save(fasta_path, save_path)

# root@autodl-container-15ef4cab54-d89cf4cf:~/ATG# python /root/ATG/feature_extraction/get_esm2_feature.py
# /root/ATG/data/fastadata/Negative_test.fasta: 100 sequences loaded
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:13<00:00,  7.36it/s]
# Saved embeddings to /root/ATG/data/pretrain/Test/esm2_ne.npz
# First embedding shape: torch.Size([1280])
# Total sequences processed: 100

# /root/ATG/data/fastadata/Positive_test.fasta: 100 sequences loaded
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:04<00:00, 21.13it/s]
# Saved embeddings to /root/ATG/data/pretrain/Test/esm2_po.npz
# First embedding shape: torch.Size([1280])
# Total sequences processed: 100

# /root/ATG/data/fastadata/Negative_training.fasta: 357 sequences loaded
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [00:40<00:00,  8.73it/s]
# Saved embeddings to /root/ATG/data/pretrain/Train/esm2_ne.npz
# First embedding shape: torch.Size([1280])
# Total sequences processed: 357

# /root/ATG/data/fastadata/Positive_training.fasta: 393 sequences loaded
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 393/393 [00:19<00:00, 19.79it/s]
# Saved embeddings to /root/ATG/data/pretrain/Train/esm2_po.npz
# First embedding shape: torch.Size([1280])
# Total sequences processed: 393