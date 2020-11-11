from torch import nn
import torch
import torch.nn.functional as F
from .utils import SE_net

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activateion = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(
            squeeze_planes, expand1x1_planes, kernel_size=1
        )
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activateion(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], dim=1)

class SqueezeNet(nn.Module):
    def __init__(self, version='1_0', num_classes=8, num_dense_layer=3, num_input=3, growth_rate=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_input = num_input
        if version == '1_0':
            if self.num_input == 3:
                modules_1 = []
                modules_2 = []
                modules_3 = []
                _in_channels = 1
                for i in range(num_dense_layer):
                    modules_1.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_2.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_3.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    _in_channels += growth_rate
                self.head_spt = nn.Sequential(*modules_1)
                self.head_chr = nn.Sequential(*modules_2)
                self.head_mfc = nn.Sequential(*modules_3)
                self.x_red = nn.Conv2d(_in_channels*3, 96, kernel_size=3, stride=2)
                self.se = SE_net(96, 96)
                self.squ_in = nn.Conv2d(96, 96, kernel_size=7, stride=2)
            elif self.num_input == 7:
                modules_1 = []
                modules_2 = []
                modules_3 = []
                modules_4 = []
                modules_5 = []
                modules_6 = []
                modules_7 = []
                _in_channels = 1
                for i in range(num_dense_layer):
                    modules_1.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_2.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_3.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_4.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_5.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_6.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    modules_7.append(MakeDense(_in_channels, growth_rate=growth_rate))
                    _in_channels += growth_rate
                self.head_spt = nn.Sequential(*modules_1)
                self.head_chr = nn.Sequential(*modules_2)
                self.head_cqt = nn.Sequential(*modules_3)
                self.head_cens = nn.Sequential(*modules_4)
                self.head_mfc = nn.Sequential(*modules_5)
                self.head_mfc1d = nn.Sequential(*modules_6)
                self.head_mfc2d = nn.Sequential(*modules_7)
                self.x_red = nn.Conv2d(_in_channels*7, 96, kernel_size=3, stride=2)
                self.se = SE_net(96, 96)
                self.squ_in = nn.Conv2d(96, 96, kernel_size=7, stride=2)
            elif self.num_input == 1:
                self.squ_in = nn.Conv2d(1, 96, kernel_size=7, stride=2)
            else:
                raise ValueError("Unsupport SqueezeNet input type {input}:"
                                "1_0 or 1_1 expected".format(input=num_input))
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1.1':
            self.features = nn.Sequential(
                nn.Conv2d(7, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceol_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 48, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupport SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))
        final_conv=nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if self.num_input == 3:
            x[0] = self.head_spt(x[0])
            x[1] = self.head_chr(x[1])
            x[2] = self.head_mfc(x[2])
            x = torch.cat((x[0], x[1], x[2]), 1)
            # x = self.x_red(x)
            x = self.se(self.x_red(x))
        elif self.num_input == 7:
            x[0] = self.head_spt(x[0])
            x[1] = self.head_chr(x[1])
            x[2] = self.head_cqt(x[2])
            x[3] = self.head_cens(x[3])
            x[4] = self.head_mfc(x[4])
            x[5] = self.head_mfc1d(x[5])
            x[6] = self.head_mfc2d(x[6])
            x = torch.cat((x[0], x[1], x[2], x[3], x[4], x[5], x[6]), 1)
            x = self.se(self.x_red(x))
        x = self.squ_in(x)
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.norm_layer = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = F.relu(self.conv(x))
        
        out = self.norm_layer(out)
        out = torch.cat((x, out), 1)
        return out


def _squeezenet(version, pretrained, progress, num_input, growth_rate, **kwargs):
    model = SqueezeNet(
        version=version, num_input=num_input, growth_rate=growth_rate, **kwargs
    )
    return model


def squeezenet1_0(pretrained=False, progress=True, num_input=3, growth_rate=16, **kwargs):
    return _squeezenet('1_0', pretrained, progress, num_input, growth_rate, **kwargs)

def squeezenet1_1(pretrained=False, progress=True, num_input=3, growth_rate=16, **kwargs):
    return _squeezenet('1_1', pretrained, progress, num_input, growth_rate, **kwargs)