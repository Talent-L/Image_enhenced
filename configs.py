"""
DIST (DiT-based) 配置文件
基因表达超分辨网络的训练与推理参数
"""


class Config:
    """DIST DiT 网络配置类"""

    # === 训练基础参数 ===
    epoch = 200
    batch_size = 128
    learning_rate = 0.001
    test_positive = True  # 若为 True，输出会被裁剪为非负值

    # === DiT 网络结构参数 ===
    hidden_dim = 512       # Transformer 隐藏层维度
    num_heads = 8          # 多头注意力头数
    num_layers = 5         # DiT Block 层数
    mlp_ratio = 4.0        # MLP 隐藏层维度比例 (hidden_dim * mlp_ratio)
    dropout = 0.0          # Dropout 比例

    # === 可选扩展参数 ===
    use_cross_gene_attn = False   # 是否使用跨基因注意力
    weight_decay = 0.01          # AdamW 权重衰减
    use_scheduler = False        # 是否使用学习率调度器 (CosineAnnealing)
    grad_clip = 0.0              # 梯度裁剪阈值，0 表示不裁剪

    def __init__(self):
        pass
