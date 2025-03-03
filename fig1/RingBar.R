library(ggplot2)

# 创建数据
individual_custom <- c(
  "AAC", "DPC", "AADP", "AAC_PSSM", "DPC_PSSM", "AADP_PSSM", "ProtT5", "ESM2-650M", "ProtBert",
  "AAC", "DPC", "AADP", "AAC_PSSM", "DPC_PSSM", "AADP_PSSM", "ProtT5", "ESM2-650M", "ProtBert",
  "AAC", "DPC", "AADP", "AAC_PSSM", "DPC_PSSM", "AADP_PSSM", "ProtT5", "ESM2-650M", "ProtBert"
)
value_custom <- c(
  0.8450, 0.8450, 0.8450, 0.9700, 0.9700, 0.9750, 0.9800, 0.9850, 0.9450,
  0.8502, 0.8502, 0.8502, 0.9700, 0.9697, 0.9749, 0.9802, 0.9847, 0.9447,
  0.6917, 0.6917, 0.6917, 0.9400, 0.9402, 0.9500, 0.9602, 0.9704, 0.8900
)
df <- data.frame(
  individual = individual_custom,
  value = value_custom 
)

# 为数据添加ID列
df$id <- seq(1, nrow(df))

# 计算角度，用于合理分布标签
df$angle<-ifelse(df$id<=14,96-df$id*12,96-df$id*12+180)
df$hjust<-ifelse(df$id<=14,0.2,1)

# 为每个组分配颜色填充
df$fill <- rep(c("Accuracy", "F1_Score", "MCC"), each = 9)

# 绘制环形柱状图
p <- ggplot(df, aes(x = as.factor(id), y = value)) +
  geom_bar(stat = "identity", aes(fill = fill)) +   # 使用fill为每个分组设置颜色
  coord_polar() +                                   # 使用极坐标系，形成环形布局
  ylim(-0.6, 1.2) +                                 # 设置Y轴范围，留出空心空间
  geom_text(aes(x = id, y = value + 0.1, label = individual,  # 标签的位置和名称
                angle = angle, hjust = hjust), size = 3) +   # 标签角度和对齐方式
  theme_minimal() +                                 # 使用最小化主题，简化背景
  ylab("") + xlab("") +                             # 隐藏X轴和Y轴标签
  theme(axis.text.y = element_blank(),              # 隐藏Y轴文本
        axis.ticks.y = element_blank(),             # 隐藏Y轴刻度
        axis.text.x = element_blank(),              # 隐藏X轴文本
        panel.grid = element_blank(),               # 隐藏网格线
        legend.position = c(1, 0),                  # 将图例放在右下角
        legend.justification = c(1, 0),            
        legend.background = element_rect(color = "#D6D6D8", size = 0.5),  # 图例框的边框颜色和大小
  ) +                
  scale_fill_manual(values = c("#E6827A", "#F6B480", "#6B98C4"), 
                    labels = c("Accuracy", "F1_Score", "MCC")) +
  labs(fill = NULL)
ggsave("D:/Major/AIProject/ATG/fig2/RingBar.png", plot = p, width = 8, height = 8)

