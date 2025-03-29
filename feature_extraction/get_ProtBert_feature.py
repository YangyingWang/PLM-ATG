from transformers import BertModel, BertTokenizer
import torch
from Bio import SeqIO
import gc #垃圾回收模块
import os
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化 T5 模型和分词器
def get_ProtBERT_model():
    model =BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    model = model.to(device)
    model = model.eval()
    return model, tokenizer

# 提取序列嵌入特征
def get_embeddings(model, tokenizer, seqs):
    results = {}
    for id, seq in seqs.items():
        seq = ' '.join(list(seq))  # 将序列转化为适合ProtBERT的格式
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 平均池化以获得每个蛋白质的嵌入
            embeddings = outputs.last_hidden_state.mean(dim=1)
            results[id] = embeddings.cpu().numpy().squeeze()  # 将嵌入特征保存到结果字典中
            
        except Exception as e:
            print(f"Error processing sequence {id}: {str(e)}")
    
    return results

def get_seq(file_path):
    seqs = {} # 用于存储序列数据的列表
    with open(file_path) as file:
        for record in SeqIO.parse(file, "fasta"):
            seq = str(record.seq)
            if len(seq) > 7000:
                seq = seq[:7000]  # 截断过长的序列
            seqs[record.id] = seq
    print(f"{file_path}: {len(seqs)} sequences loaded")
    return seqs

def process_and_save(model, tokenizer, fasta_path, save_path):
    seq_dict = get_seq(fasta_path)

    # 对序列按长度进行排序（升序）
    seq_dict = dict(sorted(seq_dict.items(), key=lambda item: len(item[1])))

    embeddings = get_embeddings(model, tokenizer, seq_dict)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **embeddings)
    print(f"Saved embeddings to {save_path}")

    if embeddings:
        first_key = next(iter(embeddings))
        print(f"First embedding shape: {embeddings[first_key].shape}")
        print(f"Total sequences processed: {len(embeddings)}\n")

if __name__ == '__main__':
    model, tokenizer = get_ProtBERT_model()
    gc.collect() # 清理内存，收集垃圾

    file_paths = {
        "D:/Major/AIProject/ATG/data/pretrain/Test/bert_ne.npz":
          "D:/Major/AIProject/ATG/data/fastadata/Negative_test.fasta",

        "D:/Major/AIProject/ATG/data/pretrain/Test/bert_po.npz":
          "D:/Major/AIProject/ATG/data/fastadata/Positive_test.fasta",

        "D:/Major/AIProject/ATG/data/pretrain/Train/bert_ne.npz":
          "D:/Major/AIProject/ATG/data/fastadata/Negative_training.fasta",

        "D:/Major/AIProject/ATG/data/pretrain/Train/bert_po.npz":
         "D:/Major/AIProject/ATG/data/fastadata/Positive_training.fasta"
    }

    for save_path, fasta_path in file_paths.items():
        process_and_save(model, tokenizer, fasta_path, save_path)

# /content/drive/MyDrive/autophagy_protein/data/fastadata/Negative_test.fasta: 100 sequences loaded
# Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Test/bert_ne.npz
# First embedding shape: (1024,)
# Total sequences processed: 100

# /content/drive/MyDrive/autophagy_protein/data/fastadata/Positive_test.fasta: 100 sequences loaded
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Test/bert_po.npz
# First embedding shape: (1024,)
# Total sequences processed: 100

# /content/drive/MyDrive/autophagy_protein/data/fastadata/Negative_training.fasta: 357 sequences loaded
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Train/bert_ne.npz
# First embedding shape: (1024,)
# Total sequences processed: 357

# /content/drive/MyDrive/autophagy_protein/data/fastadata/Positive_training.fasta: 393 sequences loaded
# Saved embeddings to /content/drive/MyDrive/autophagy_protein/data/pretrain/Train/bert_po.npz
# First embedding shape: (1024,)
# Total sequences processed: 393