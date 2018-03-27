import torch
from torch import nn
import torch.nn.functional as F
from math import floor
from math import sqrt as sqrt
class SubBlock(nn.Module):
    """
    This piece of the DenseBlock receives an input feature map
    x and transforms it through a dense, composite function H(x).

    The transformation H(x) is a composition of 3 consecutive 
    operations: BN - ReLU - Conv (3x3).

    In the bottleneck variant of the SubBlock, a 1x1 conv is
    added to the transformation function H(x), reducing the number
    of input feature maps and improving computational efficiency.
    """
    def __init__(self, in_channels, out_channels, bottleneck, p):
        """
        Initialize the different parts of the SubBlock.

        Params
        ------
        - in_channels: number of input channels in the convolution.
        - out_channels: number of output channels in the convolution.
        - bottleneck: if true, applies the bottleneck variant of H(x).
        - p: if greater than 0, applies dropout after the convolution.
        """
        super(SubBlock, self).__init__()
        self.bottleneck = bottleneck
        self.p = p

        in_channels_2 = in_channels
        out_channels_2 = out_channels

        if bottleneck:
            in_channels_1 = in_channels
            out_channels_1 = out_channels * 4
            in_channels_2 = out_channels_1

            self.bn1 = nn.BatchNorm2d(in_channels_1)
            self.conv1 = nn.Conv2d(in_channels_1,
                                   out_channels_1,
                                   kernel_size=1)
            print ('bottle_neck')
        self.bn2 = nn.BatchNorm2d(in_channels_2)
        self.conv2 = nn.Conv2d(in_channels_2, 
                               out_channels_2, 
                               kernel_size=3, 
                               padding=1)
    def forward(self, x):
        """
        Compute the forward pass of the composite transformation H(x),
        where x is the concatenation of the current and all preceding
        feature maps.
        """
        if self.bottleneck:
            out = self.conv1(F.relu(self.bn1(x)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)
            out = self.conv2(F.relu(self.bn2(out)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)
        else:
            out = self.conv2(F.relu(self.bn2(x)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)  
        return torch.cat((x, out), 1)

class DenseBlock(nn.Module):
    """
    Block that connects L layers directly with each other in a 
    feed-forward fashion.

    Concretely, this block is composed of L SubBlocks sharing a 
    common growth rate k (Figure 1 in the paper).
    """
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, p):
        """
        Initialize the different parts of the DenseBlock.

        Params
        ------
        - num_layers: the number of layers L in the dense block.
        - in_channels: the number of input channels feeding into the first 
          subblock.
        - growth_rate: the number of output feature maps produced by each subblock.
          This number is common across all subblocks.
        """
        super(DenseBlock, self).__init__()

        # create L subblocks
        layers = []
        for i in range(num_layers):
            cumul_channels = in_channels + i * growth_rate
            layers.append(SubBlock(cumul_channels, growth_rate, bottleneck, p))

        self.block = nn.Sequential(*layers)
        self.out_channels = cumul_channels + growth_rate

    def forward(self, x):
        """
        Feed the input feature map x through the L subblocks 
        of the DenseBlock.
        """
        out = self.block(x)
        return out

class TransitionLayer(nn.Module):
    """
    This layer is placed between consecutive Dense blocks. 
    It allows the network to downsample the size of feature 
    maps using the pooling operator.

    Concretely, this layer is a composition of 3 operations:
    BN - Conv (1x1) - AveragePool
    
    Additionally, this layer can perform compression by reducing
    the number of output feature maps using a compression factor
    theta.
    """
    def __init__(self, in_channels, theta, p):
        """
        Initialize the different parts of the TransitionBlock.

        Params
        ------
        - in_channels: number of input channels.
        - theta: compression factor in the range [0, 1]. Set to 0.5
          in the paper when using DenseNet-BC.
        """
        super(TransitionLayer, self).__init__()
        self.p = p
        self.out_channels = int(floor(theta*in_channels))

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, 
                              self.out_channels, 
                              kernel_size=1)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        out = self.pool(self.conv(F.relu(self.bn(x))))
        if self.p > 0:
            out = F.dropout(out, p=self.p, training=self.training)
        return out
