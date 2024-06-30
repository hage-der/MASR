import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd, _size_2_t, Union, _pair, Tensor, Optional


class Conv2dValid(_ConvNd):
    """
    Conv2d operator for VALID mode padding.
    """

    def __init__(
            self,
            in_channels: int,   # 输入图像通道数
            out_channels: int,  # 卷积产生的通道数
            kernel_size: _size_2_t,  # 卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核
            stride: _size_2_t = 1,   # 卷积步长，默认为1。可以设为1个int型数或者一个(int, int)型的元组。
            padding: Union[str, _size_2_t] = 0,  # 填充操作，控制padding_mode的数目。简言之，就是决定图像边沿填充的方式
            dilation: _size_2_t = 1,             # 扩张操作：控制kernel点（卷积核点）的间距，默认值:1
            groups: int = 1,                     # group参数的作用是控制分组卷积，默认不分组，为1组。输入图像通道数
            bias: bool = True,                   # 为真，则在输出中添加一个可学习的偏差。默认：True
            padding_mode: str = 'zeros',  # padding模式，默认为Zero-padding
            device=None,
            dtype=None,
            valid_trigx: bool = False,
            valid_trigy: bool = False
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2dValid, self).__init__(
            in_channels, out_channels, kernel_size_,
            stride_, padding_, dilation_, False, _pair(0),
            groups, bias, padding_mode, **factory_kwargs)
        self.valid_trigx = valid_trigx
        self.valid_trigy = valid_trigy

    def _conv_forward(
            self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        validx, validy = 0, 0
        if self.valid_trigx:
            validx = (input.size(-2) * (self.stride[-2] - 1) - 1 + self.kernel_size[-2]) // 2
        if self.valid_trigy:
            validy = (input.size(-1) * (self.stride[-1] - 1) - 1 + self.kernel_size[-1]) // 2
        return F.conv2d(input, weight, bias, self.stride, (validx, validy), self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
