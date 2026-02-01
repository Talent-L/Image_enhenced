"""
DiT工具函数
包含位置编码、损失函数等辅助功能
"""

import torch
import torch.nn as nn
import numpy as np
import math


def get_2d_sincos_pos_embed(embed_dim, h, w):
    """
    生成2D正弦余弦位置编码
    
    Args:
        embed_dim: 嵌入维度
        h: 高度
        w: 宽度
    
    Returns:
        pos_embed: (H*W, embed_dim) 位置编码
    """
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (w, h) -> (h, w)
    grid = np.stack(grid, axis=0)  # (2, h, w)
    grid = grid.reshape([2, 1, h, w])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed  # (H*W, embed_dim)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从网格生成2D位置编码
    """
    assert embed_dim % 2 == 0
    
    # 分别对h和w使用一半的维度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, base_scale=10000):
    """
    从1D位置生成正弦余弦编码
    
    Args:
        embed_dim: 嵌入维度
        pos: 位置数组 (M,)
    
    Returns:
        emb: (M, embed_dim) 位置编码
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / base_scale**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionEmbedding2D(nn.Module):
    """
    可学习的2D位置编码，支持动态尺寸
    使用正弦余弦初始化
    """
    def __init__(self, hidden_dim, max_h=64, max_w=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_h = max_h
        self.max_w = max_w
        
        # 初始化为正弦余弦编码
        pos_embed = get_2d_sincos_pos_embed(hidden_dim, max_h, max_w)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float()) 
        
    def forward(self, h, w):
        """
        获取指定尺寸的位置编码
        
        Args:
            h: 高度
            w: 宽度
            
        Returns:
            pos_embed: (H*W, hidden_dim)
        """
        # 从预计算的编码中提取对应尺寸
        pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, h, w)
        return torch.from_numpy(pos_embed).float().to(self.pos_embed.device)


class GeneEmbedding(nn.Module):
    """
    基因表达嵌入模块
    将每个spot的基因表达值投影到hidden space
    """
    def __init__(self, num_genes, hidden_dim):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        
        # 使用MLP进行嵌入
        self.embed = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, C) 每个位置的基因表达值，C是基因数
        Returns:
            embedded: (B, N, hidden_dim)
        """
        return self.embed(x)


class OutputHead(nn.Module):
    """
    输出头：从token embedding预测4个位置的基因表达
    4个位置对应2x2块: (0,0), (0,1), (1,0), (1,1)
    """
    def __init__(self, hidden_dim, num_genes):
        super().__init__()
        self.num_genes = num_genes
        
        # 分别预测4个位置
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4 * num_genes),  # 输出4个位置
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, hidden_dim)
        Returns:
            out: (B, N, 4, num_genes) - 4个位置的基因表达预测
        """
        B, N, _ = x.shape
        out = self.head(x)  # (B, N, 4*C)
        out = out.reshape(B, N, 4, self.num_genes)
        return out


def dit_loss(pred, hr, in_tissue):
    """
    计算DiT超分损失
    网络预测4个位置，全部参与损失计算

    Args:
        pred: (B, C, 4, H, W) - 预测的4个位置 (0,0),(0,1),(1,0),(1,1)
        hr: (B, C, 2*H, 2*W) - 高分辨率ground truth
        in_tissue: (B, 2*H, 2*W) - 组织mask

    Returns:
        loss: 标量损失值
    """
    loss = 0

    # pred[:, :, 0] 对应位置 (0, 0) - hr中的 (:, :, 0::2, 0::2)
    # pred[:, :, 1] 对应位置 (0, 1) - hr中的 (:, :, 0::2, 1::2)
    # pred[:, :, 2] 对应位置 (1, 0) - hr中的 (:, :, 1::2, 0::2)
    # pred[:, :, 3] 对应位置 (1, 1) - hr中的 (:, :, 1::2, 1::2)

    loss += ((pred[:, :, 0] - hr[:, :, 0::2, 0::2])**2 * in_tissue[:, None, 0::2, 0::2]).mean()
    loss += ((pred[:, :, 1] - hr[:, :, 0::2, 1::2])**2 * in_tissue[:, None, 0::2, 1::2]).mean()
    loss += ((pred[:, :, 2] - hr[:, :, 1::2, 0::2])**2 * in_tissue[:, None, 1::2, 0::2]).mean()
    loss += ((pred[:, :, 3] - hr[:, :, 1::2, 1::2])**2 * in_tissue[:, None, 1::2, 1::2]).mean()

    return loss


def reconstruct_hr(pred):
    """
    将预测的4个位置组合成完整的高分辨率输出
    全部4个位置由网络预测，不再使用输入

    Args:
        pred: (B, C, 4, H, W) - 预测的4个位置 (0,0),(0,1),(1,0),(1,1)

    Returns:
        hr_out: (B, C, 2*H, 2*W) - 完整的高分辨率输出
    """
    B, C, _, H, W = pred.shape
    hr_out = torch.zeros(B, C, 2*H, 2*W, device=pred.device, dtype=pred.dtype)

    hr_out[:, :, 0::2, 0::2] = pred[:, :, 0]  # (0, 0)
    hr_out[:, :, 0::2, 1::2] = pred[:, :, 1]  # (0, 1)
    hr_out[:, :, 1::2, 0::2] = pred[:, :, 2]  # (1, 0)
    hr_out[:, :, 1::2, 1::2] = pred[:, :, 3]  # (1, 1)

    return hr_out
