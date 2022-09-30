import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ClsModule(nn.Module):
    def __init__(self, num_classes):
        super(ClsModule, self).__init__()

        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 9 * self.num_classes, kernel_size=3, padding=1),
                                      nn.Sigmoid()
                                      )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, self.num_classes)

        return x


class RegModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 4 * 9, 3, padding=1),
                                      )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()   # convert B x C x W x H to B x W x H x C
        x = x.view(batch_size, -1, 4)
        return x


# for backbone
class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet50_list = nn.ModuleList(list(resnet50(pretrained=pretrained).children())[:-2])  # to layer 1

    def forward(self, x):

        x = self.resnet50_list[0](x)  # 7 x 7 conv 64
        x = self.resnet50_list[1](x)  # bn
        x = self.resnet50_list[2](x)  # relu
        x = self.resnet50_list[3](x)  # 3 x 3 maxpool

        x = self.resnet50_list[4](x)       # layer 1   # [B,  256, w/4, h/4]
        c3 = x = self.resnet50_list[5](x)  # layer 2   # [B,  512, w/8, h/8]
        c4 = x = self.resnet50_list[6](x)  # layer 3   # [B, 1024, w/16, h/16]
        c5 = x = self.resnet50_list[7](x)  # layer 4   # [B, 2048, w/32, h/32]

        return [c3, c4, c5]


class FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stride = 128

        channels = [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize()

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.lateral3.apply(init_layer)
        self.lateral4.apply(init_layer)
        self.lateral5.apply(init_layer)
        self.pyramid6.apply(init_layer)
        self.pyramid7.apply(init_layer)
        self.smooth3.apply(init_layer)
        self.smooth4.apply(init_layer)
        self.smooth5.apply(init_layer)

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, size=p4.size()[2:]) + p4  # interpolate p5 to p4's w h           # nearest
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, size=p3.size()[2:]) + p3  # interpolate p4 to p3's w h           # nearest

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return [p3, p4, p5, p6, p7]
