import numpy as np
import matplotlib.pyplot as plt

labels = ['Acc', 'Pre', 'Sen', 'Spe', 'F1-score', 'Mcc']
data1 = [0.905, 0.904, 0.890, 0.920, 0.918, 0.810]
data2 = [0.945, 0.959, 0.930, 0.960, 0.944, 0.890]
data3 = [0.995, 0.990, 0.995, 0.990, 0.995, 0.990]

# 将第一个数据点连接回最后一个
data1 += data1[:1]
data2 += data2[:1]
data3 += data3[:1]

# 角度计算
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 使图形闭合

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.plot(angles, data1, color='#2C91E0', linewidth=2, label="ATGPred-FL")
ax.fill(angles, data1, color='#2C91E0', alpha=0.35)
ax.plot(angles, data2, color='#3ABF99', linewidth=2, label="EnsembleDL-ATG")
ax.fill(angles, data2, color='#3ABF99', alpha=0.35)
ax.plot(angles, data3, color='#F0A73A', linewidth=2, label="PLMs-ATG")
ax.fill(angles, data3, color='#F0A73A', alpha=0.35)

ax.set_yticklabels([])  # 隐藏径向的刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)  # 调整标签字体大小
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('D:/Major/AIProject/ATG/fig2/RadChart.png')
plt.show()