import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')

labels = ['ProtT5', 'ESM2-650M', 'ProtBert']
accuracy = [0.9800, 0.9850, 0.9450]
f1_score = [0.9802, 0.9847, 0.9447]
mcc = [0.9602, 0.9704, 0.8900]
accuracy_err = [0.015, 0.01, 0.02]
f1_score_err = [0.01, 0.01, 0.02]
mcc_err = [0.023, 0.013, 0.035]

# 设置位置和宽度
x = np.arange(len(labels))
width = 0.9

fig, axs = plt.subplots(1, 3, figsize=(9, 6))
colors = ['#2C91E0', '#3ABF99', '#F0A73A']

# 设置每个子图的背景为灰色
for ax in axs:
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', alpha=0.7)

# Accuracy
for i in range(len(labels)):
    axs[0].bar(x[i] - width, accuracy[i], yerr=accuracy_err[i], capsize=5, color=colors[i], width=width)
    axs[0].text(x[i] - width, 0.4, f'{accuracy[i]:.4f}', ha='center', va='center', color='white', fontsize=15)

axs[0].set_ylim(0, 1)
axs[0].set_xticks([])
axs[0].set_xticklabels([])
axs[0].set_title('Accuracy')

# F1-Score
for i in range(len(labels)):
    axs[1].bar(x[i], f1_score[i], yerr=f1_score_err[i], capsize=5, color=colors[i], width=width)
    axs[1].text(x[i], 0.4, f'{f1_score[i]:.4f}', ha='center', va='center', color='white', fontsize=15)

axs[1].set_ylim(0, 1)
axs[1].set_xticks([])
axs[1].set_xticklabels([])
axs[1].set_title('F1-Score')

# MCC
for i in range(len(labels)):
    axs[2].bar(x[i] + width, mcc[i], yerr=mcc_err[i], capsize=5, color=colors[i], width=width)
    axs[2].text(x[i] + width, 0.4, f'{mcc[i]:.4f}', ha='center', va='center', color='white', fontsize=15)

axs[2].set_ylim(0, 1)
axs[2].set_xticks([])
axs[2].set_xticklabels([])
axs[2].set_title('MCC')

fig.legend(labels, loc='lower center', ncol=3)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('D:/Major/AIProject/ATG/fig2/PLMs.png')
plt.show()