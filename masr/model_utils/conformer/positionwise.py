# -*- coding: utf-8 -*-
import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """原来的前馈
    Positionwise feed forward layer."""

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Module = nn.ReLU()):
        """Construct a PositionwiseFeedForward object.

        FeedForward are appied on each position of the sequence.
        The output dim is same with the input dim.

        Args:
            idim (int): Input dimenstion.
            hidden_units (int): The number of hidden units.
            dropout_rate (float): Dropout rate.
            activation (torch.nn.Layer): Activation function
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            xs: input tensor (B, Lmax, D)
        Returns:
            output tensor, (B, Lmax, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ElderlyVoiceAdaptiveFeedForward(nn.Module):
    """针对老年人语音识别优化的前馈层。"""

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Module = nn.ReLU(),
                 frequency_enhancement: bool = True,
                 temporal_adaptation: bool = True,
                 articulation_boost: float = 0.15):
        super(ElderlyVoiceAdaptiveFeedForward, self).__init__()

        # 基本前馈结构
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

        # 老年人语音特定增强
        self.frequency_enhancement = frequency_enhancement
        if frequency_enhancement:
            self.freq_attention = nn.Sequential(
                nn.Linear(idim, idim),
                nn.Sigmoid()
            )
            with torch.no_grad():
                bias_values = torch.linspace(0.1, 0.3, idim)
                self.freq_attention[0].bias.copy_(bias_values)

        self.temporal_adaptation = temporal_adaptation
        if temporal_adaptation:
            self.tempo_gate = nn.Sequential(
                nn.Linear(idim, idim),
                nn.Tanh()
            )

        self.articulation_boost = articulation_boost
        if articulation_boost > 0:
            self.clarity_factors = nn.Parameter(torch.ones(idim) * articulation_boost)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # 标准前馈处理
        intermediate = self.activation(self.w_1(xs))
        intermediate = self.dropout(intermediate)
        output = self.w_2(intermediate)

        # 应用老年人语音增强
        if self.frequency_enhancement:
            freq_weights = self.freq_attention(xs)
            output = output * (1.0 + freq_weights)

        if self.temporal_adaptation:
            tempo_factor = self.tempo_gate(xs)
            output = output + (output * tempo_factor * 0.2)

        if self.articulation_boost > 0:
            clarity_boost = torch.tanh(output) * self.clarity_factors
            output = output + clarity_boost

        return output