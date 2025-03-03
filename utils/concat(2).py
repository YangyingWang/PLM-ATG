import numpy as np

test_t5_files = [
    'D:/Major/AIProject/ATG/data/pretrain/Test/t5_ne.npz',
    'D:/Major/AIProject/ATG/data/pretrain/Test/t5_po.npz'
]
test_esm2_files = [
    'D:/Major/AIProject/ATG/data/pretrain/Test/esm2_ne.npz',
    'D:/Major/AIProject/ATG/data/pretrain/Test/esm2_po.npz'
]
train_t5_files = [
    'D:/Major/AIProject/ATG/data/pretrain/Train/t5_ne.npz', 
    'D:/Major/AIProject/ATG/data/pretrain/Train/t5_po.npz'
] 
train_esm2_files = [
    'D:/Major/AIProject/ATG/data/pretrain/Train/esm2_ne.npz',
    'D:/Major/AIProject/ATG/data/pretrain/Train/esm2_po.npz'
]

def load_data(paths):
    X, y, ids = [], [], []
    
    for path in paths:
        with np.load(path) as data:
            features = [data[key] for key in data]
            X.extend(features)
            ids.extend([key for key in data])
            if '_ne' in path:
                y.extend([0] * len(features))
            elif '_po' in path:
                y.extend([1] * len(features))
            else:
                print(f"Warning: Unrecognized file pattern in {path}")

    return np.array(X), np.array(y), ids

def concat(t5_files, esm2_files, output_file):
    print("Start load T5 data...")
    X_t5, y_t5, ids_t5 = load_data(t5_files)
    print("X_t5 shape:", X_t5.shape)
    print("y_t5 shape:", y_t5.shape)

    print("Start load ESM2 data...")
    X_esm2, y_esm2, ids_esm2 = load_data(esm2_files)
    print("X_esm2 shape:", X_esm2.shape)
    print("y_esm2 shape:", y_esm2.shape)

    # 获取 T5 和 ESM2 的公共样本 ID
    common_ids = set(ids_t5).intersection(ids_esm2)
    print(f"Common IDs found: {len(common_ids)}")

    # 构建ID到索引的映射
    t5_id_to_index = {id_: i for i, id_ in enumerate(ids_t5)}
    esm2_id_to_index = {id_: i for i, id_ in enumerate(ids_esm2)}

    # 提取共有ID的特征并拼接
    X_concat, y_concat = [], []
    for id_ in common_ids:
        t5_index = t5_id_to_index[id_]
        esm2_index = esm2_id_to_index[id_]
        
        # 拼接特征
        concat_features = np.concatenate([X_t5[t5_index], X_esm2[esm2_index]])
        X_concat.append(concat_features)
        
        # 保持标签一致
        y_concat.append(y_t5[t5_index])

    X_concat = np.array(X_concat)
    y_concat = np.array(y_concat)
    print(f"X_concat shape: {X_concat.shape}")
    print(f"y_concat shape: {y_concat.shape}")
    
    # 保存拼接后的特征
    np.savez(output_file, features=X_concat, labels=y_concat)
    print(f"拼接完成并保存到 {output_file}")

concat(train_t5_files, train_esm2_files, 'D:/Major/AIProject/ATG/data/pretrain/Train/t5_esm2.npz')
concat(test_t5_files, test_esm2_files, 'D:/Major/AIProject/ATG/data/pretrain/Test/t5_esm2.npz')

# 检查结果
data = np.load('D:/Major/AIProject/ATG/data/pretrain/Train/t5_esm2.npz')
print("文件中包含的键:", data.keys())
labels = data['labels']
num_po = np.sum(labels == 1)
num_ne = np.sum(labels == 0)
print(f"正样本数量: {num_po}")
print(f"负样本数量: {num_ne}")
print(f"总样本数量: {len(labels)}")

# Start load T5 data...
# X_t5 shape: (750, 1024)
# y_t5 shape: (750,)
# Start load ESM2 data...
# X_esm2 shape: (750, 1280)
# y_esm2 shape: (750,)
# Common IDs found: 750
# X_concat shape: (750, 2304)
# y_concat shape: (750,)
# 拼接完成并保存到 D:/Major/AIProject/ATG/data/pretrain/Train/t5_esm2.npz
# Start load T5 data...
# X_t5 shape: (200, 1024)
# y_t5 shape: (200,)
# Start load ESM2 data...
# X_esm2 shape: (200, 1280)
# y_esm2 shape: (200,)
# Common IDs found: 200
# X_concat shape: (200, 2304)
# y_concat shape: (200,)
# 拼接完成并保存到 D:/Major/AIProject/ATG/data/pretrain/Test/t5_esm2.npz

# 文件中包含的键: KeysView(<numpy.lib.npyio.NpzFile object at 0x000001F7C9407AF0>)
# 正样本数量: 100
# 负样本数量: 100
# 总样本数量: 200

# 文件中包含的键: KeysView(<numpy.lib.npyio.NpzFile object at 0x000001F0B3D07AF0>)
# 正样本数量: 393
# 负样本数量: 357
# 总样本数量: 750