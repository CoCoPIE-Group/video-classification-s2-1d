import math
import torch.nn as nn
from torchvision.models.video.resnet import VideoResNet,BasicBlock,Conv2Plus1D,R2Plus1dStem,Bottleneck


class VideoResNetPlus(VideoResNet):
    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400, width = 64,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], int(width*2), layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], int(width*4), layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], int(width*8), layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(int(width*8), num_classes)

        # init weights
        # self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


def r2plus1d_scaling(num_classes,multiplier=8):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    block = BasicBlock
    conv_makers = [Conv2Plus1D] * 4
    layers = [int(math.floor(multiplier/8))+1] * 4
    stem = R2Plus1dStem
    width = 8*multiplier
    model = VideoResNetPlus(block = block,conv_makers=conv_makers,layers=layers,stem=stem,width = width,num_classes=num_classes)
    return model