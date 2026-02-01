"""
DiT-based Network for Gene Expression Super-Resolution
基于DiT Block的基因表达超分辨网络

核心思想：
- 每个空间位置(spot)作为一个token
- Token = 基因表达embedding + 位置embedding
- 通过Transformer处理token间的关系
- 输出每个token对应的4个位置的基因表达 (2x2块全部预测)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from dit_utils import (
    PositionEmbedding2D,
    GeneEmbedding,
    OutputHead,
    dit_loss,
    reconstruct_hr,
    get_2d_sincos_pos_embed,
)


class DiTBlock(nn.Module):
    """
    DiT风格的Transformer Block
    包含自注意力和MLP，使用Pre-Norm结构
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 自注意力层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # MLP层
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, hidden_dim) token序列
        Returns:
            x: (B, N, hidden_dim) 处理后的token序列
        """
        # Pre-norm Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class CrossGeneAttention(nn.Module):
    """
    跨基因注意力模块
    在基因维度上进行注意力计算，捕捉基因间的相关性
    
    这里我们将每个位置的基因表达展开，让不同基因之间互相attend
    """
    def __init__(self, hidden_dim, num_genes, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_genes = num_genes
        
        # 基因级别的attention
        # 将hidden_dim拆分为num_genes个部分
        self.gene_dim = hidden_dim // num_genes if hidden_dim >= num_genes else hidden_dim
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        
    def forward(self, x):
        """
        Args:
            x: (B, N, hidden_dim)
        Returns:
            x: (B, N, hidden_dim)
        """
        B, N, D = x.shape
        
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = (D // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return x + out


class DiTEncoder(nn.Module):
    """
    DiT Encoder：将输入的基因表达转换为token表示
    """
    def __init__(self, num_genes, hidden_dim):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        
        # 基因表达嵌入
        self.gene_embed = GeneEmbedding(num_genes, hidden_dim)
        
        # 位置嵌入
        self.pos_embed = PositionEmbedding2D(hidden_dim)
        
    def forward(self, x, h, w):
        """
        Args:
            x: (B, N, C) 基因表达，N=H*W, C=num_genes
            h, w: 空间维度
        Returns:
            tokens: (B, N, hidden_dim)
        """
        # 基因表达嵌入
        tokens = self.gene_embed(x)  # (B, N, hidden_dim)
        
        # 添加位置嵌入
        pos_embed = self.pos_embed(h, w)  # (N, hidden_dim)
        tokens = tokens + pos_embed.unsqueeze(0)
        
        return tokens


class DiTDecoder(nn.Module):
    """
    DiT Decoder：从token表示解码出4个位置的基因表达
    """
    def __init__(self, hidden_dim, num_genes):
        super().__init__()
        self.output_head = OutputHead(hidden_dim, num_genes)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, hidden_dim)
        Returns:
            out: (B, N, 4, num_genes)
        """
        return self.output_head(x)


class DiTNet(nn.Module):
    """
    完整的DiT网络
    
    架构:
    1. Encoder: 基因表达 + 位置 -> token embedding
    2. DiT Blocks: token间的自注意力 + MLP
    3. Decoder: token -> 4个位置的基因表达预测
    """
    def __init__(self, conf, num_genes):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_dim = getattr(conf, 'hidden_dim', 256)
        self.num_heads = getattr(conf, 'num_heads', 8)
        self.num_layers = getattr(conf, 'num_layers', 4)
        self.mlp_ratio = getattr(conf, 'mlp_ratio', 4.0)
        self.dropout = getattr(conf, 'dropout', 0.0)
        
        # Encoder
        self.encoder = DiTEncoder(num_genes, self.hidden_dim)
        
        # DiT Blocks (Spatial Attention)
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_dim, 
                self.num_heads, 
                self.mlp_ratio,
                self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # 可选：跨基因注意力层（在DiT blocks之间）
        self.use_cross_gene_attn = getattr(conf, 'use_cross_gene_attn', False)
        if self.use_cross_gene_attn:
            self.cross_gene_attn = nn.ModuleList([
                CrossGeneAttention(self.hidden_dim, num_genes, self.num_heads)
                for _ in range(self.num_layers)
            ])
        
        # Decoder
        self.decoder = DiTDecoder(self.hidden_dim, num_genes)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, lr):
        """
        Args:
            lr: (B, C, H, W) 低分辨率基因表达图像
                C: 基因数
                H, W: 空间维度
        Returns:
            pred: (B, C, 4, H, W) 预测的4个位置 (0,0),(0,1),(1,0),(1,1)
        """
        B, C, H, W = lr.shape
        N = H * W

        # 转换为token序列: (B, C, H, W) -> (B, N, C)
        x = lr.permute(0, 2, 3, 1).reshape(B, N, C)

        # Encoder: 得到token embeddings
        tokens = self.encoder(x, H, W)  # (B, N, hidden_dim)

        # DiT Blocks
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if self.use_cross_gene_attn:
                tokens = self.cross_gene_attn[i](tokens)

        # Decoder: 预测4个位置，全部由网络预测
        out = self.decoder(tokens)  # (B, N, 4, C)

        # 重排为 (B, C, 4, H, W)
        out = out.permute(0, 3, 2, 1)  # (B, C, 4, N)
        out = out.reshape(B, C, 4, H, W)

        return out


class Net:
    """
    训练和测试的封装类，与原接口兼容
    """
    def __init__(self, train_set, test_set, conf, device=None):
        self.conf = conf
        
        # 设备控制
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")
        
        # 数据
        self.train_lr, self.train_hr, self.train_in_tissue = train_set
        self.test_set = test_set
        self.num_genes = self.train_lr.shape[0]
        print(f"[INFO] Number of genes: {self.num_genes}")
        
        # 设置默认的DiT参数
        if not hasattr(conf, 'hidden_dim'):
            conf.hidden_dim = 512
        if not hasattr(conf, 'num_heads'):
            conf.num_heads = 8
        if not hasattr(conf, 'num_layers'):
            conf.num_layers = 5
        if not hasattr(conf, 'mlp_ratio'):
            conf.mlp_ratio = 4.0
        if not hasattr(conf, 'dropout'):
            conf.dropout = 0.0
        if not hasattr(conf, 'use_cross_gene_attn'):
            conf.use_cross_gene_attn = False
            
        print(f"[INFO] DiT Config: hidden_dim={conf.hidden_dim}, "
              f"num_heads={conf.num_heads}, num_layers={conf.num_layers}")
        
        # 模型
        self.model = DiTNet(conf, self.num_genes).to(self.device)
        self.opt = optim.AdamW(
            self.model.parameters(), 
            lr=conf.learning_rate,
            weight_decay=getattr(conf, 'weight_decay', 0.01)
        )
        
        # 学习率调度器（可选）
        self.use_scheduler = getattr(conf, 'use_scheduler', False)
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=conf.epoch
            )
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Total parameters: {total_params:,}")
        
    def run(self):
        self.train()
        return self.test()
    
    def train(self):
        self.model.train()
        
        print("Start training DiT model...")
        for epoch in range(self.conf.epoch):
            # 准备数据
            lr = torch.tensor(
                self.train_lr, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, C, H, W)
            hr = torch.tensor(
                self.train_hr, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, C, 2H, 2W)
            in_tissue = torch.tensor(
                self.train_in_tissue, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, 2H, 2W)
            
            # 前向传播
            pred = self.model(lr)  # (1, C, 4, H, W)
            
            # 计算损失
            loss = dit_loss(pred, hr, in_tissue)
            
            # 反向传播
            self.opt.zero_grad()
            loss.backward()
            
            # 梯度裁剪（可选）
            if hasattr(self.conf, 'grad_clip') and self.conf.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.conf.grad_clip
                )
            
            self.opt.step()
            
            if self.use_scheduler:
                self.scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.conf.epoch}] Loss: {loss.item():.6f}")
    
    def test(self):
        self.model.eval()
        
        with torch.no_grad():
            lr = torch.tensor(
                self.test_set, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, C, H, W)
            
            pred = self.model(lr)  # (1, C, 4, H, W)

            # 重构完整的高分辨率输出
            hr_out = reconstruct_hr(pred)  # (1, C, 2H, 2W)
            hr_out = hr_out.squeeze(0)  # (C, 2H, 2W)
            
            if self.conf.test_positive:
                hr_out[hr_out < 0] = 0
                
        return hr_out.cpu().numpy()
