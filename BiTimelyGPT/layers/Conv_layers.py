import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from layers.snippets import Transpose

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class TemporalConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(TemporalConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)


class Conv1dSubampling(nn.Module):
    """
    Convolutional 1d subsampling with padding to control sequence length reduction.
    Args:
        in_channels (int): Number of channels in the input (e.g., n_mels for spectrogram)
        out_channels (int): Number of channels produced by the convolution (typically model dimension)
        reduce_time_layers (int): Number of halving conv layers to apply (default is 2 for 1/4 reduction)

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns:
        - **outputs** (batch, time, dim): Tensor produced by the convolution
    """
    def __init__(self, in_channels: int, out_channels: int, reduce_time_layers: int = 2) -> None:
        super(Conv1dSubampling, self).__init__()

        # First, reduce the time_length
        time_reduce_layers = []
        for _ in range(reduce_time_layers):
            time_reduce_layers.extend([
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU()
            ])
        self.time_reduce = nn.Sequential(*time_reduce_layers)

        # Then, mix the model_dim
        self.dim_mix = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, inputs: Tensor) -> (Tensor, Tensor):
        inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        tokens = self.time_reduce(inputs)
        outputs = self.dim_mix(tokens)
        outputs = outputs.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return outputs, tokens.permute(0, 2, 1)


class Conv1dUpsampling(nn.Module):
    def __init__(self, hidden_dim: int, reduce_time_layers: int = 2):
        super(Conv1dUpsampling, self).__init__()

        # Upsample only in the time dimension, increase time dimensions of the hidden_states tensor
        layers = []
        for _ in range(reduce_time_layers):
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU()
            ])
        self.time_upsample = nn.Sequential(*layers)

        # Reduce the potential effects of padded artifacts introduced by the upsampling
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        x = self.time_upsample(x)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return x


# # Generate random input data
# batch_size = 24
# seq_length = 256
# in_channels = 4
# out_channels = 512
#
# inputs = torch.rand(batch_size, seq_length, in_channels)
# input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
# print(inputs.shape, input_lengths.shape)
#
# # Initialize the PaddedConvSubampling module
# subsampling_layer = Conv1dSubampling_new(in_channels=in_channels, out_channels=out_channels)
#
# # Pass the input through the PaddedConvSubampling module
# subsampling_output, subsampling_output_lengths = subsampling_layer(inputs)
# print(f"Output shape after subsampling: {subsampling_output.shape}")