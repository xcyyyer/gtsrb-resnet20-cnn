"""CIFAR-style ResNet-20: 3 stages x 3 BasicBlocks, option-A shortcuts.
Adaptation: 43 output classes, RGB input resized to 32x32.
DS replaces both 3x3 convolutions in every residual block; stem unchanged.
SE: after second BN, before residual addition. No hidden extra BN/ReLU in DS.
"""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('baseline', 'ds', 'se', 'ds_se')

class OptionA(nn.Module):
    def forward(self, x):
        x = x[:, :, ::2, ::2]
        return F.pad(x, (0, 0, 0, 0, x.shape[1] // 2, x.shape[1] // 2))

class Separable(nn.Sequential):
    def __init__(self, ci, co, stride):
        super().__init__(nn.Conv2d(ci, ci, 3, stride, 1, groups=ci, bias=False),
                         nn.Conv2d(ci, co, 1, bias=False))

class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = max(1, c // 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(),
                                nn.Conv2d(h, c, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.pool(x))

class Block(nn.Module):
    def __init__(self, ci, co, stride, ds=False, se=False):
        super().__init__()
        def conv(a, b, s):
            return Separable(a, b, s) if ds else nn.Conv2d(a, b, 3, s, 1, bias=False)
        self.conv1, self.bn1 = conv(ci, co, stride), nn.BatchNorm2d(co)
        self.conv2, self.bn2 = conv(co, co, 1), nn.BatchNorm2d(co)
        self.attention = SE(co) if se else nn.Identity()
        self.shortcut = nn.Identity() if stride == 1 else OptionA()
    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.attention(self.bn2(self.conv2(z)))
        return F.relu(z + self.shortcut(x))

class ResNet20(nn.Module):
    def __init__(self, variant='baseline', classes=43):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        ds, se = variant in ('ds', 'ds_se'), variant in ('se', 'ds_se')
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU())
        blocks, ci = [], 16
        for stage, co in enumerate((16, 32, 64)):
            for i in range(3):
                stride = 2 if stage > 0 and i == 0 else 1
                blocks.append(Block(ci, co, stride, ds, se))
                ci = co
        self.blocks = nn.Sequential(*blocks)
        self.pool, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(64, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))

def profile(model, size=32):
    """Conv/Linear MACs only; one multiply-accumulate = 1 MAC (not 2 FLOPs).
    Excludes BN, activations, pooling, additions, SE feature multiplication.
    """
    macs, handles = [0], []
    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            macs[0] += out.numel() * m.in_features
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    training = model.training
    model.eval()
    with torch.no_grad(): model(torch.zeros(1, 3, size, size))
    for h in handles: h.remove()
    model.train(training)
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'conv_linear_macs': macs[0], 'input': [1, 3, size, size],
            'mac_scope': 'Conv2d and Linear only; 1 multiply-accumulate = 1 MAC'}
