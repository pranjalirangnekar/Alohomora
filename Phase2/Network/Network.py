"""
#BasicCNN
import torch.nn as nn
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#ImprovedCNN

#ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Adding the residual (skip) connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

#ResNext
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(BasicBlock, self).__init__()

        # Grouped convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Adding the residual (skip) connection
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality=32, num_classes=10):
        super(ResNeXt, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cardinality=cardinality)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cardinality=cardinality)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cardinality=cardinality)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cardinality=cardinality)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, cardinality):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, cardinality))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, cardinality=cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNeXt29_32x4d():
    return ResNeXt(BasicBlock, [3, 3, 3, 3], cardinality=32)
"""
#DenseNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """
    DenseNet bottleneck layer.
    """
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        bottleneck_output = self.conv1(F.relu(self.bn1(x)))
        bottleneck_output = self.conv2(F.relu(self.bn2(bottleneck_output)))
        return torch.cat([x, bottleneck_output], 1)


class Transition(nn.Module):
    """
    Transition layer to reduce feature map size and channels.
    """
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseBlock(nn.Module):
    """
    Dense block consisting of multiple bottleneck layers.
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10, growth_rate=32, block_layers=(6, 12, 24, 16)):
        super(DenseNet121, self).__init__()
        self.num_classes = num_classes
        self.growth_rate = growth_rate

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Dense blocks and transition layers
        self.in_channels = 2 * growth_rate
        self.dense1 = DenseBlock(block_layers[0], self.in_channels, growth_rate)
        self.in_channels += block_layers[0] * growth_rate
        self.trans1 = Transition(self.in_channels, self.in_channels // 2)
        self.in_channels //= 2

        self.dense2 = DenseBlock(block_layers[1], self.in_channels, growth_rate)
        self.in_channels += block_layers[1] * growth_rate
        self.trans2 = Transition(self.in_channels, self.in_channels // 2)
        self.in_channels //= 2

        self.dense3 = DenseBlock(block_layers[2], self.in_channels, growth_rate)
        self.in_channels += block_layers[2] * growth_rate
        self.trans3 = Transition(self.in_channels, self.in_channels // 2)
        self.in_channels //= 2

        self.dense4 = DenseBlock(block_layers[3], self.in_channels, growth_rate)
        self.in_channels += block_layers[3] * growth_rate

        # Classification layer
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)

        x = F.relu(self.bn2(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


