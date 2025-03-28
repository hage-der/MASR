# -*- coding: utf-8 -*-
import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from masr.model_utils.conformer.attention import MultiHeadedAttention

class HierarchicalMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head, n_feat, dropout_rate, local_window_size=32,
                 local_heads=None, fusion_type="adaptive",
                 # 新增参数
                 num_bands=4):  # <-- 新增参数，表示频带数量
        super().__init__(n_head, n_feat, dropout_rate)

        # 初始化参数
        self.n_feat = n_feat
        self.d_k = n_feat // n_head
        self.local_heads = local_heads if local_heads is not None else n_head // 2
        self.global_heads = n_head - self.local_heads
        self.local_window_size = local_window_size

        # 新增：频带数量参数
        self.num_bands = num_bands  # <-- 新增

        # 计算局部和全局特征的维度
        self.local_dim = self.local_heads * self.d_k
        self.global_dim = self.global_heads * self.d_k

        # 调整投影层的维度
        self.local_projection = nn.Linear(self.local_dim, n_feat)
        self.global_projection = nn.Linear(self.global_dim, n_feat)

        # 融合相关的初始化保持不变...
        self.fusion_type = fusion_type
        if fusion_type == "fixed":
            self.register_buffer('fusion_weights', torch.tensor([0.6, 0.4]))
            self.fusion_module = None
        elif fusion_type == "adaptive":
            self.fusion_module = nn.Sequential(
                nn.Linear(n_feat * 2, n_feat),
                nn.Tanh(),
                nn.Linear(n_feat, 2),
                nn.Softmax(dim=-1)
            )
        # 新增：频谱感知动态融合
        elif fusion_type == "spectrum_aware":  # <-- 新增融合类型
            # 频谱分析器
            self.spectrum_analyzer = nn.Sequential(  # <-- 新增
                nn.Linear(n_feat, self.num_bands),
                nn.Sigmoid()
            )

            # 频带特定融合网络
            self.fusion_module = nn.Sequential(  # <-- 修改
                nn.Linear(n_feat * 2, n_feat),
                nn.Tanh(),
                nn.Linear(n_feat, self.num_bands * 2),  # 为每个频带生成权重对
                nn.Softmax(dim=-1)
            )
        else:  # "learned"
            self.fusion_module = nn.Sequential(
                nn.Linear(n_feat * 2, n_feat),
                nn.LayerNorm(n_feat),
                nn.ReLU(),
                nn.Linear(n_feat, n_feat * 2),
                nn.Sigmoid()
            )

        self.fusion_norm = nn.LayerNorm(n_feat)

    def _compute_local_attention(self, q, k, v, mask):
        """计算局部注意力并确保输出维度正确"""
        batch_size, _, time_len, _ = q.size()

        # 选择局部注意力的头
        q_local = q[:, :self.local_heads]
        k_local = k[:, :self.local_heads]
        v_local = v[:, :self.local_heads]

        # 创建并应用局部掩码
        local_mask = self._create_local_mask(time_len, self.local_window_size, q.device)
        scores = torch.matmul(q_local, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask.shape[2] > 0:
            pad_mask = mask.unsqueeze(1).eq(0)
            if pad_mask.dim() == 3:
                pad_mask = pad_mask.unsqueeze(2)
            final_mask = pad_mask | local_mask
            scores = scores.masked_fill(final_mask, -float('inf'))
        else:
            scores = scores.masked_fill(local_mask, -float('inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        local_out = torch.matmul(attn, v_local)
        # 重塑维度以便于后续处理
        local_out = local_out.transpose(1, 2).reshape(batch_size, time_len, -1)
        # 投影到原始特征维度
        return self.local_projection(local_out)

    def _create_local_mask(self, seq_length: int, window_size: int,
                           device: torch.device) -> torch.Tensor:
        """创建局部注意力的掩码矩阵

        Args:
            seq_length (int): 序列长度
            window_size (int): 局部窗口大小
            device (torch.device): 设备类型

        Returns:
            torch.Tensor: 布尔类型的掩码矩阵，形状为 (1, 1, seq_length, seq_length)
            True 表示该位置被掩蔽，False 表示该位置可以参与注意力计算

        注意:
            - 对于每个位置i，只允许它注意到范围[i-window_size//2, i+window_size//2]内的位置
            - 掩码矩阵是对称的
            - 会自动处理序列边界情况
        """
        # 创建位置索引矩阵
        positions = torch.arange(seq_length, device=device)
        # 计算每对位置之间的距离
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        # 创建窗口掩码：如果距离超过窗口大小的一半，则掩蔽
        window_mask = torch.abs(distances) > (window_size // 2)
        # 添加批次和头部维度
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        return window_mask

    def _compute_global_attention(self, q, k, v, mask):
        """计算全局注意力并确保输出维度正确"""
        batch_size, _, time_len, _ = q.size()

        # 选择全局注意力的头
        q_global = q[:, self.local_heads:]
        k_global = k[:, self.local_heads:]
        v_global = v[:, self.local_heads:]

        scores = torch.matmul(q_global, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask.shape[2] > 0:
            scores = scores.masked_fill(mask.unsqueeze(1).eq(0), -float('inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        global_out = torch.matmul(attn, v_global)
        # 重塑维度以便于后续处理
        global_out = global_out.transpose(1, 2).reshape(batch_size, time_len, -1)
        # 投影到原始特征维度
        return self.global_projection(global_out)

    def forward(self, query, key, value, mask=torch.ones([0, 0, 0], dtype=torch.bool),
                pos_emb=torch.empty([0]), cache=torch.zeros([0, 0, 0, 0])):
        """确保维度正确的前向传播"""
        q, k, v = self.forward_qkv(query, key, value)

        if cache.shape[0] > 0:
            key_cache, value_cache = torch.split(cache, cache.shape[-1] // 2, dim=-1)
            k = torch.concat([key_cache, k], dim=2)
            v = torch.concat([value_cache, v], dim=2)
        new_cache = torch.concat((k, v), dim=-1)

        # 计算局部和全局注意力（现在它们都会返回正确的维度）
        local_features = self._compute_local_attention(q, k, v, mask)
        global_features = self._compute_global_attention(q, k, v, mask)

        # 融合特征
        if self.fusion_type == "fixed":
            output = self.fusion_weights[0] * local_features + self.fusion_weights[1] * global_features
        elif self.fusion_type == "adaptive":
            weights = self.fusion_module(torch.cat([local_features, global_features], dim=-1))
            output = weights[..., 0:1] * local_features + weights[..., 1:2] * global_features
        # 新增：频谱感知动态融合
        elif self.fusion_type == "spectrum_aware":  # <-- 新增融合逻辑
            batch_size, seq_len, _ = local_features.shape

            # 1. 提取频谱信息
            spectrum_info = self.spectrum_analyzer(torch.mean(query, dim=1))  # [B, num_bands]

            # 2. 生成融合权重
            fusion_weights = self.fusion_module(torch.cat([local_features, global_features], dim=-1))
            # 重塑为每个频带一对权重 [B, T, num_bands*2] -> [B, T, num_bands, 2]
            fusion_weights = fusion_weights.view(batch_size, seq_len, self.num_bands, 2)

            # 3. 将特征重塑为频带表示
            local_bands = local_features.view(batch_size, seq_len, self.num_bands, -1)
            global_bands = global_features.view(batch_size, seq_len, self.num_bands, -1)

            # 4. 根据频谱信息调整融合权重
            adjusted_weights = fusion_weights * spectrum_info.unsqueeze(1).unsqueeze(-1)
            # 归一化确保权重和为1
            adjusted_weights = F.normalize(adjusted_weights, p=1, dim=-1)

            # 5. 频带级融合
            output_bands = (
                    local_bands * adjusted_weights[..., 0:1] +
                    global_bands * adjusted_weights[..., 1:2]
            )

            # 6. 重塑回原始维度
            output = output_bands.reshape(batch_size, seq_len, -1)
        else:  # "learned"
            combined = torch.cat([local_features, global_features], dim=-1)
            output = self.fusion_module(combined)

        output = self.fusion_norm(output)
        return output, new_cache

# class HierarchicalMultiHeadedAttention(MultiHeadedAttention):
#     def __init__(self, n_head, n_feat, dropout_rate, local_window_size=32,
#                  local_heads=None, fusion_type="adaptive"):
#         super().__init__(n_head, n_feat, dropout_rate)
#
#         # 初始化参数
#         self.n_feat = n_feat
#         self.d_k = n_feat // n_head
#         self.local_heads = local_heads if local_heads is not None else n_head // 2
#         self.global_heads = n_head - self.local_heads
#         self.local_window_size = local_window_size
#
#         # 计算局部和全局特征的维度
#         self.local_dim = self.local_heads * self.d_k
#         self.global_dim = self.global_heads * self.d_k
#
#         # 调整投影层的维度
#         self.local_projection = nn.Linear(self.local_dim, n_feat)
#         self.global_projection = nn.Linear(self.global_dim, n_feat)
#
#         # 融合相关的初始化保持不变...
#         self.fusion_type = fusion_type
#         if fusion_type == "fixed":
#             self.register_buffer('fusion_weights', torch.tensor([0.6, 0.4]))
#             self.fusion_module = None
#         elif fusion_type == "adaptive":
#             self.fusion_module = nn.Sequential(
#                 nn.Linear(n_feat * 2, n_feat),
#                 nn.Tanh(),
#                 nn.Linear(n_feat, 2),
#                 nn.Softmax(dim=-1)
#             )
#         else:  # "learned"
#             self.fusion_module = nn.Sequential(
#                 nn.Linear(n_feat * 2, n_feat),
#                 nn.LayerNorm(n_feat),
#                 nn.ReLU(),
#                 nn.Linear(n_feat, n_feat * 2),
#                 nn.Sigmoid()
#             )
#
#         self.fusion_norm = nn.LayerNorm(n_feat)
#
#     def _compute_local_attention(self, q, k, v, mask):
#         """计算局部注意力并确保输出维度正确"""
#         batch_size, _, time_len, _ = q.size()
#
#         # 选择局部注意力的头
#         q_local = q[:, :self.local_heads]
#         k_local = k[:, :self.local_heads]
#         v_local = v[:, :self.local_heads]
#
#         # 创建并应用局部掩码
#         local_mask = self._create_local_mask(time_len, self.local_window_size, q.device)
#         scores = torch.matmul(q_local, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
#
#         if mask.shape[2] > 0:
#             pad_mask = mask.unsqueeze(1).eq(0)
#             if pad_mask.dim() == 3:
#                 pad_mask = pad_mask.unsqueeze(2)
#             final_mask = pad_mask | local_mask
#             scores = scores.masked_fill(final_mask, -float('inf'))
#         else:
#             scores = scores.masked_fill(local_mask, -float('inf'))
#
#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#
#         local_out = torch.matmul(attn, v_local)
#         # 重塑维度以便于后续处理
#         local_out = local_out.transpose(1, 2).reshape(batch_size, time_len, -1)
#         # 投影到原始特征维度
#         return self.local_projection(local_out)
#
#     def _create_local_mask(self, seq_length: int, window_size: int,
#                            device: torch.device) -> torch.Tensor:
#         """创建局部注意力的掩码矩阵
#
#         Args:
#             seq_length (int): 序列长度
#             window_size (int): 局部窗口大小
#             device (torch.device): 设备类型
#
#         Returns:
#             torch.Tensor: 布尔类型的掩码矩阵，形状为 (1, 1, seq_length, seq_length)
#             True 表示该位置被掩蔽，False 表示该位置可以参与注意力计算
#
#         注意:
#             - 对于每个位置i，只允许它注意到范围[i-window_size//2, i+window_size//2]内的位置
#             - 掩码矩阵是对称的
#             - 会自动处理序列边界情况
#         """
#         # 创建位置索引矩阵
#         positions = torch.arange(seq_length, device=device)
#         # 计算每对位置之间的距离
#         distances = positions.unsqueeze(1) - positions.unsqueeze(0)
#         # 创建窗口掩码：如果距离超过窗口大小的一半，则掩蔽
#         window_mask = torch.abs(distances) > (window_size // 2)
#         # 添加批次和头部维度
#         window_mask = window_mask.unsqueeze(0).unsqueeze(0)
#         return window_mask
#
#     def _compute_global_attention(self, q, k, v, mask):
#         """计算全局注意力并确保输出维度正确"""
#         batch_size, _, time_len, _ = q.size()
#
#         # 选择全局注意力的头
#         q_global = q[:, self.local_heads:]
#         k_global = k[:, self.local_heads:]
#         v_global = v[:, self.local_heads:]
#
#         scores = torch.matmul(q_global, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
#
#         if mask.shape[2] > 0:
#             scores = scores.masked_fill(mask.unsqueeze(1).eq(0), -float('inf'))
#
#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#
#         global_out = torch.matmul(attn, v_global)
#         # 重塑维度以便于后续处理
#         global_out = global_out.transpose(1, 2).reshape(batch_size, time_len, -1)
#         # 投影到原始特征维度
#         return self.global_projection(global_out)
#
#     def forward(self, query, key, value, mask=torch.ones([0, 0, 0], dtype=torch.bool),
#                 pos_emb=torch.empty([0]), cache=torch.zeros([0, 0, 0, 0])):
#         """确保维度正确的前向传播"""
#         q, k, v = self.forward_qkv(query, key, value)
#
#         if cache.shape[0] > 0:
#             key_cache, value_cache = torch.split(cache, cache.shape[-1] // 2, dim=-1)
#             k = torch.concat([key_cache, k], dim=2)
#             v = torch.concat([value_cache, v], dim=2)
#         new_cache = torch.concat((k, v), dim=-1)
#
#         # 计算局部和全局注意力（现在它们都会返回正确的维度）
#         local_features = self._compute_local_attention(q, k, v, mask)
#         global_features = self._compute_global_attention(q, k, v, mask)
#
#         # 融合特征
#         if self.fusion_type == "fixed":
#             output = self.fusion_weights[0] * local_features + self.fusion_weights[1] * global_features
#         elif self.fusion_type == "adaptive":
#             weights = self.fusion_module(torch.cat([local_features, global_features], dim=-1))
#             output = weights[..., 0:1] * local_features + weights[..., 1:2] * global_features
#         else:  # "learned"
#             combined = torch.cat([local_features, global_features], dim=-1)
#             output = self.fusion_module(combined)
#
#         output = self.fusion_norm(output)
#         return output, new_cache