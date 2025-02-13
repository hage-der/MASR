# -*- coding: utf-8 -*-
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from masr.model_utils.conformer.attention import MultiHeadedAttention


# 首先定义配置类，它将帮助我们管理所有的参数
@dataclass
class AttentionConfig:
    """注意力机制的配置类"""
    n_head: int  # 注意力头的数量
    n_feat: int  # 输入特征的维度
    dropout_rate: float  # dropout比率
    reduction_factor: int = 4  # 特征降维因子，用于控制计算复杂度
    scales: Tuple[int] = (1, 4, 8)  # 时间尺度因子，用于捕捉不同长度的依赖关系
    energy_threshold: float = 0.1  # 能量阈值，用于判断语音片段的重要性
    use_relative_pos: bool = True  # 是否使用相对位置编码


class MultiscaleMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """初始化多尺度注意力层

        这个初始化方法首先调用父类的初始化，然后设置自己的特有属性。这样可以确保
        我们既继承了父类的功能，又添加了自己的特性。

        Args:
            n_head: 注意力头的数量
            n_feat: 输入特征的维度
            dropout_rate: dropout比率
        """
        # 首先调用父类的初始化方法
        super().__init__(n_head, n_feat, dropout_rate)

        # 存储重要的类属性
        self.head_size = n_head  # 保存注意力头数量
        self.feature_size = n_feat  # 保存特征维度

        # 设置缩放参数
        self.scales = (1, 4, 8)  # 默认的时间尺度
        self.reduction_factor = 4  # 默认的降维因子

        # 计算降维后的特征维度，确保能被注意力头数整除
        self.reduced_dim = (n_feat // self.reduction_factor // n_head) * n_head
        self.d_k = self.reduced_dim // n_head  # 每个注意力头的维度

        # 创建特征降维层
        self.feature_reduction = nn.Sequential(
            nn.Linear(self.feature_size, self.reduced_dim),
            nn.ReLU(),
            nn.LayerNorm(self.reduced_dim)
        )

        # 创建特征升维层
        self.feature_expansion = nn.Sequential(
            nn.Linear(self.reduced_dim, self.feature_size),
            nn.LayerNorm(self.feature_size)
        )

        # 创建注意力的线性变换层
        self.linear_q = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.linear_k = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.linear_v = nn.Linear(self.reduced_dim, self.reduced_dim)

        # 创建高效的时间池化层
        self.pooling_layers = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Conv1d(
                    in_channels=self.reduced_dim,
                    out_channels=self.reduced_dim,
                    kernel_size=scale,
                    stride=scale,
                    groups=self.reduced_dim,
                    padding=(scale - 1) // 2
                ),
                nn.BatchNorm1d(self.reduced_dim),
                nn.ReLU()
            )
            for scale in self.scales if scale > 1
        })

        # 创建能量感知门控
        self.energy_gate = nn.Sequential(
            nn.Linear(1, self.reduced_dim),
            nn.Sigmoid()
        )

    def _compute_energy_gating(self, x: torch.Tensor) -> torch.Tensor:
        """计算输入序列的能量门控值"""
        energy = torch.norm(x, dim=-1, keepdim=True)
        return self.energy_gate(energy)

    def _temporal_pooling(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """执行时间维度上的池化操作"""
        if scale == 1:
            return x

        batch_size, time_len, feat_dim = x.size()

        # 处理填充
        pad_len = (scale - (time_len % scale)) % scale
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.transpose(1, 2)
        x = self.pooling_layers[str(scale)](x)
        return x.transpose(1, 2)

    def _interpolate(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """将张量插值到目标长度"""
        if x.size(1) == target_len:
            return x

        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.transpose(1, 2)

    def forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理查询、键和值张量，实现多尺度特征提取

        这个方法执行以下步骤：
        1. 特征降维以减少计算量
        2. 在多个时间尺度上处理特征
        3. 使用能量门控机制关注重要特征
        4. 合并不同尺度的特征

        Args:
            query: 查询张量 [batch, time1, size]
            key: 键张量 [batch, time2, size]
            value: 值张量 [batch, time2, size]

        Returns:
            处理后的查询、键、值张量，准备用于注意力计算
        """
        batch_size = query.size(0)
        original_time = key.size(1)

        # 特征降维
        q_reduced = self.feature_reduction(query)
        k_reduced = self.feature_reduction(key)
        v_reduced = self.feature_reduction(value)

        # 计算能量门控
        energy_gate = self._compute_energy_gating(query)

        # 多尺度处理
        scale_outputs = []
        for scale in self.scales:
            q = self._temporal_pooling(q_reduced * energy_gate, scale)
            k = self._temporal_pooling(k_reduced * energy_gate, scale)
            v = self._temporal_pooling(v_reduced * energy_gate, scale)

            # 插值回原始长度
            q = self._interpolate(q, query.size(1))
            k = self._interpolate(k, original_time)
            v = self._interpolate(v, original_time)

            # 线性变换
            q = self.linear_q(q)
            k = self.linear_k(k)
            v = self.linear_v(v)

            scale_outputs.append((q, k, v))

        # 计算注意力权重
        attention_weights = []
        for q, _, _ in scale_outputs:
            score = torch.mean(q * energy_gate, dim=-1)
            attention_weights.append(score)

        # 堆叠并进行softmax
        attention_weights = torch.stack(attention_weights, dim=0)
        weights = F.softmax(attention_weights, dim=0)

        # 合并多尺度特征
        q_combined = torch.zeros_like(scale_outputs[0][0])
        k_combined = torch.zeros_like(scale_outputs[0][1])
        v_combined = torch.zeros_like(scale_outputs[0][2])

        for i, ((q, k, v), w) in enumerate(zip(scale_outputs, weights)):
            w = w.unsqueeze(-1)
            q_combined += q * w
            k_combined += k * w
            v_combined += v * w

        # 调整维度用于多头注意力计算
        q = q_combined.view(batch_size, -1, self.head_size, self.d_k).transpose(1, 2)
        k = k_combined.view(batch_size, -1, self.head_size, self.d_k).transpose(1, 2)
        v = v_combined.view(batch_size, -1, self.head_size, self.d_k).transpose(1, 2)

        return q, k, v

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pos_emb: Optional[torch.Tensor] = None,
            cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播函数

        Args:
            query: 查询张量 [batch, time1, size]
            key: 键张量 [batch, time2, size]
            value: 值张量 [batch, time2, size]
            mask: 注意力掩码 [batch, time1, time2]
            pos_emb: 位置编码
            cache: 注意力缓存

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 输出张量 [batch, time1, size]
                - 更新后的缓存
        """
        q, k, v = self.forward_qkv(query, key, value)

        # 处理缓存
        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        # 创建新的缓存
        new_cache = torch.cat([k, v], dim=-1)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # 计算输出
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.reduced_dim)

        # 特征升维
        output = self.feature_expansion(x)

        return output, new_cache