from typing import Tuple

import torch
from torch import nn
from typeguard import check_argument_types


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
       Conformer 模型中的 ConvolutionModule
    """

    def __init__(self,
                 channels: int,                        # 卷积层的通道数
                 kernel_size: int = 15,                # 卷积层的内核大小
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,                 # 是否使用因果卷积
                 bias: bool = True,
                 adaptive_scale: bool = False,
                 init_weights: bool = False):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        assert check_argument_types()
        super().__init__()
        self.bias = bias
        self.channels = channels
        self.kernel_size = kernel_size
        self.adaptive_scale = adaptive_scale
        self.ada_scale = torch.nn.Parameter(torch.ones([1, 1, channels]), requires_grad=adaptive_scale)
        self.ada_bias = torch.nn.Parameter(torch.zeros([1, 1, channels]), requires_grad=adaptive_scale)

        self.pointwise_conv1 = nn.Conv1d(channels,
                                         2 * channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=bias)
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        # self.lorder 用于区分是否是因果卷积，如果 self.lorder > 0：是因果卷积，输入将在前向左侧填充 self.lorder 帧。 else: 这是一个对称卷积
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            # 对于非因果卷积，kernel_size 应该是奇数
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(channels,
                                        channels,
                                        kernel_size,
                                        stride=1,
                                        padding=padding,
                                        groups=channels,
                                        bias=bias)

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(channels,
                                         channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=bias)
        self.activation = activation
        if init_weights:
            self.init_weights()

    def init_weights(self):
        pw_max = self.channels ** -0.5
        dw_max = self.kernel_size ** -0.5
        torch.nn.init.uniform_(self.pointwise_conv1.weight.data, -pw_max, pw_max)
        if self.bias:
            torch.nn.init.uniform_(self.pointwise_conv1.bias.data, -pw_max, pw_max)
        torch.nn.init.uniform_(self.depthwise_conv.weight.data, -dw_max, dw_max)
        if self.bias:
            torch.nn.init.uniform_(self.depthwise_conv.bias.data, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pointwise_conv2.weight.data, -pw_max, pw_max)
        if self.bias:
            torch.nn.init.uniform_(self.pointwise_conv2.bias.data, -pw_max, pw_max)

    def forward(
            self,
            x: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        计算卷积模块
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        if self.adaptive_scale:
            x = self.ada_scale * x + self.ada_bias
        # exchange the temporal dimension and the feature dimension
        # 交换时间维度和特征维度
        x = x.transpose(1, 2)  # (#batch, channels, time)
        # mask batch padding
        # 掩码批量填充
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # Pointwise Conv
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        # GLU mechanism
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        # 使用两个卷积一同训练，并将结果相加
        temp = x
        x1 = self.depthwise_conv(temp)
        x2 = self.depthwise_conv(temp)
        x = x1 + x2
        # BatchNorm
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        #     激活函数
        x = self.activation(self.norm(x))

        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache
