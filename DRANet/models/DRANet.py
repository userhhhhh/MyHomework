import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from .batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools
from functools import partial
import torchvision.transforms as ttransforms
from torchvision.models import resnet18

USE_NEW_Discriminator = False
USE_NEW_Generator = True
USE_D_OPTIMIZATION = False

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Separator(nn.Module):
    def __init__(self, imsize, converts, ch=64, down_scale=2):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
        )
        self.w = nn.ParameterDict()
        w, h = imsize
        for cv in converts:
            self.w[cv] = nn.Parameter(torch.ones(1, ch, h//down_scale, w//down_scale), requires_grad=True)

    def forward(self, features, converts=None):
        contents, styles = dict(), dict()
        for key in features.keys():
            styles[key] = self.conv(features[key])  # equals to F - wS(F) see eq.(2)
            contents[key] = features[key] - styles[key]  # equals to wS(F)
            if '2' in key:  # for 3 datasets: source-mid-target
                source, target = key.split('2')
                contents[target] = contents[key]

        if converts is not None:  # separate features of converted images to compute consistency loss.
            for cv in converts:
                source, target = cv.split('2')
                contents[cv] = self.w[cv] * contents[source]
        return contents, styles

if USE_NEW_Generator == False and USE_NEW_Discriminator == False:
    class Generator(nn.Module):
        def __init__(self, channels=512):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(True),
                ResidualBlock(32, 32),
                ResidualBlock(32, 32),
                spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.Tanh()
            )

        def forward(self, content, style):
            return self.model(content+style)

elif USE_NEW_Discriminator == True and USE_NEW_Generator == False:
    class Generator(nn.Module):
        def __init__(self, channels=512):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(True),

                ResidualBlock(64, 64),
                ResidualBlock(64, 64),
                ResidualBlock(64, 64),  # 新增一层 residual 增强表达

                spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)),  # 上采样到更高分辨率
                nn.ReLU(True),

                spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(True),

                spectral_norm(nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)),
                nn.Tanh()
            )

        def forward(self, content, style):
            return self.model(content + style)

else:
    class Generator(nn.Module):
        def __init__(self, channels=512):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),  # 保持分辨率
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                ResidualBlock(128, 128),
                ResidualBlock(128, 128),

                spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                spectral_norm(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)),
                nn.Tanh()
            )

        def forward(self, content, style):
            return self.model(content + style)


class Classifier(nn.Module):
    def __init__(self, channels=3, num_classes=10):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(12288, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out


class Discriminator_USPS(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_USPS, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

if USE_NEW_Discriminator == False:
    class Discriminator_MNIST(nn.Module):
        def __init__(self, channels=3):
            super(Discriminator_MNIST, self).__init__()
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.ReLU(True)
            )
            self.fc = nn.Sequential(
                nn.Linear(256*4*4, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            output = self.conv(x)
            output = output.view(output.size(0),-1)
            output = self.fc(output)
            return output

else:
    class Discriminator_MNIST(nn.Module):
        def __init__(self):
            super(Discriminator_MNIST, self).__init__()

            # 基于 ResNet18 的浅层特征提取器（不加载预训练）
            base = resnet18(pretrained=False)
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

            self.feature_extractor = nn.Sequential(
                base.conv1, base.bn1, base.relu,
                base.layer1,  # 输出: [B, 64, 32, 32]
                base.layer2   # 输出: [B, 128, 32, 32]
            )

            # 更深的卷积网络（新增多层）
            self.extra_conv = nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),  # -> [B, 256, 16, 16]
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(256, 256, 3, stride=1, padding=1)),  # -> [B, 256, 16, 16]
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1)),  # -> [B, 512, 8, 8]
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(512, 512, 3, stride=1, padding=1)),  # -> [B, 512, 8, 8]
                nn.ReLU(True),
                spectral_norm(nn.Conv2d(512, 512, 3, stride=2, padding=1)),  # -> [B, 512, 4, 4]
                nn.ReLU(True),
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),      # -> [B, 512, 1, 1]
                nn.Flatten(),                      # -> [B, 512]
                spectral_norm(nn.Linear(512, 1)),  # -> [B, 1]
                nn.Sigmoid()
            )

        def forward(self, x):
            feat = self.feature_extractor(x)   # -> [B, 128, 32, 32]
            feat = self.extra_conv(feat)       # -> [B, 512, 2, 2] -> GAP -> [B, 512]
            out = self.classifier(feat)        # -> [B, 1]
            return out


# class Discriminator_MNIST(nn.Module):
#     def __init__(self):
#         super(Discriminator_MNIST, self).__init__()
#         # 加载 ResNet18，不加载预训练权重
#         base = resnet18(pretrained=False)

#         # 修改第一个卷积层，适配 MNIST/MNIST-M 输入分辨率
#         base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         base.maxpool = nn.Identity()  # 移除最大池化，保留更多空间信息

#         # 只保留卷积部分作为特征提取器
#         self.feature_extractor = nn.Sequential(
#             base.conv1,
#             base.bn1,
#             base.relu,
#             base.layer1,
#             base.layer2
#         )

#         # 全局平均池化 + 判别器输出
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             spectral_norm(nn.Linear(128, 1)),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         feat = self.feature_extractor(x)  # [B, 512, H', W']
#         out = self.classifier(feat)       # [B, 1]
#         return out

class PatchGAN_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(PatchGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)

