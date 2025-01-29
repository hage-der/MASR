# Author : ZY 
# Time : 2025/1/29 9:39 
# ÄÚÈÝ :
import math
from typing import Tuple

import torch
from torch import nn

from masr.model_utils.conformer.attention import MultiHeadedAttention


class MultiscaleMultiHeadedAttention(MultiHeadedAttention):
    """Multi-scale Multi-Head Attention layer for Conformer."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, scales: Tuple[int] = (1, 2, 4)):
        """Construct a MultiscaleMultiHeadedAttention object.

        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
            scales (tuple): Temporal scales for multi-scale processing.
        """
        super().__init__(n_head, n_feat, dropout_rate)
        self.scales = scales

        # Additional projections for each scale
        self.scale_projections = nn.ModuleList([
            nn.Linear(n_feat, n_feat) for _ in range(len(scales))
        ])

        # Scale weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_weights_softmax = nn.Softmax(dim=0)

    def _temporal_pooling(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Apply temporal pooling at a given scale.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size)
            scale (int): Pooling scale factor

        Returns:
            torch.Tensor: Pooled tensor (#batch, time/scale, size)
        """
        if scale == 1:
            return x

        batch_size, time_len, feat_dim = x.size()
        # Ensure the time dimension is divisible by scale
        pad_len = (scale - (time_len % scale)) % scale
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            time_len = time_len + pad_len

        # Reshape and pool
        x = x.view(batch_size, time_len // scale, scale, feat_dim)
        x = x.mean(dim=2)
        return x

    def _interpolate(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate temporal dimension to target length.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size)
            target_len (int): Target sequence length

        Returns:
            torch.Tensor: Interpolated tensor (#batch, target_len, size)
        """
        batch_size, time_len, feat_dim = x.size()
        if time_len == target_len:
            return x

        x = x.transpose(1, 2)  # (#batch, size, time)
        x = torch.nn.functional.interpolate(x, size=target_len, mode='linear')
        return x.transpose(1, 2)  # (#batch, target_len, size)

    def forward_qkv(self,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value with multi-scale processing.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size)
            key (torch.Tensor): Key tensor (#batch, time2, size)
            value (torch.Tensor): Value tensor (#batch, time2, size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed tensors
        """
        n_batch = query.size(0)
        original_time = key.size(1)

        # Process each scale
        q_scales, k_scales, v_scales = [], [], []
        for i, scale in enumerate(self.scales):
            # Apply temporal pooling
            q_scaled = self._temporal_pooling(query, scale)
            k_scaled = self._temporal_pooling(key, scale)
            v_scaled = self._temporal_pooling(value, scale)

            # Apply scale-specific projection
            q_scaled = self.scale_projections[i](q_scaled)
            k_scaled = self.scale_projections[i](k_scaled)
            v_scaled = self.scale_projections[i](v_scaled)

            # Interpolate back to original length
            q_scaled = self._interpolate(q_scaled, query.size(1))
            k_scaled = self._interpolate(k_scaled, original_time)
            v_scaled = self._interpolate(v_scaled, original_time)

            # Transform to multi-head format
            q = self.linear_q(q_scaled).view(n_batch, -1, self.h, self.d_k)
            k = self.linear_k(k_scaled).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(v_scaled).view(n_batch, -1, self.h, self.d_k)

            q_scales.append(q)
            k_scales.append(k)
            v_scales.append(v)

        # Combine scales with learned weights
        scale_weights = self.scale_weights_softmax(self.scale_weights)
        q = sum([w * q_s for w, q_s in zip(scale_weights, q_scales)])
        k = sum([w * k_s for w, k_s in zip(scale_weights, k_scales)])
        v = sum([w * v_s for w, v_s in zip(scale_weights, v_scales)])

        # Final transpose for attention computation
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v