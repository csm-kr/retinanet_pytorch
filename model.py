from torchvision.models import resnet50
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

# we refer to https://github.com/NVIDIA/retinanet-examples repo


class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.resnet50 = resnet50(pretrained=pretrained)
        self.resnet50_list = nn.ModuleList(list(self.resnet50.children())[:-2])  # to layer 1
        self.res50 = nn.Sequential(*self.resnet50_list)

    def forward(self, x):

        x = self.resnet50_list[0](x)  # 7 x 7 conv 64
        x = self.resnet50_list[1](x)  # bn
        x = self.resnet50_list[2](x)  # relu
        x = self.resnet50_list[3](x)  # 3 x 3 maxpool

        x = self.resnet50_list[4](x)  # layer 1        # [B,  256, w/4, h/4]
        c3 = x = self.resnet50_list[5](x)  # layer 2   # [B,  512, w/8, h/8]
        c4 = x = self.resnet50_list[6](x)  # layer 3   # [B, 1024, w/16, h/16]
        c5 = x = self.resnet50_list[7](x)  # layer 4   # [B, 2048, w/32, h/32]

        return [c3, c4, c5]


class FPN(nn.Module):
    '''
    Feature Pyramid Network - https://arxiv.org/abs/1612.03144
    refer to https://github.com/NVIDIA/retinanet-examples/blob/master/retinanet/model.py
    '''
    def __init__(self, baseline=Resnet50()):
        super().__init__()

        self.stride = 128
        self.baseline = baseline

        channels = [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

        # init is default -> he_uniform
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py line 114 ~ 119

        self.initialize()

    def initialize(self):
        # initialize FPN except for baseline
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.kaiming_normal_(layer.weight)
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

    def forward(self, x):
        c3, c4, c5 = self.baseline(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, size=p4.size()[2:], mode='bilinear') + p4  # interpolate p5 to p4's w h
        # p4 = F.interpolate(p5, size=p4.size()[2:]) + p4  # interpolate p5 to p4's w h
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, size=p3.size()[2:], mode='bilinear') + p3  # interpolate p4 to p3's w h
        # p3 = F.interpolate(p4, size=p3.size()[2:]) + p3  # interpolate p4 to p3's w h

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return [p3, p4, p5, p6, p7]


class RetinaNet(nn.Module):
    def __init__(self, fpn=FPN(Resnet50(pretrained=True)), num_classes=20):
        super(RetinaNet, self).__init__()

        self.fpn = fpn
        self.num_classes = num_classes
        self.cls_module = ClsModule(self.num_classes)
        self.reg_module = RegModule()

        self.initialize_subsets()
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_subsets(self):
        i = 0
        for c in self.cls_module.features.children():

            if isinstance(c, nn.Conv2d):
                if i == 8:  # final layer

                    pi = 0.01
                    b = - math.log((1 - pi) / pi)
                    nn.init.constant_(c.bias, b)
                    nn.init.normal_(c.weight, std=0.01)

                else:
                    nn.init.normal_(c.weight, std=0.01)
                    nn.init.constant_(c.bias, 0)

            i += 1

        for c in self.reg_module.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    # https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/model.py
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        features = self.fpn(inputs)  # [p3, p4, p5, p6, p7 ]
        reg = torch.cat([self.reg_module(feature) for feature in features], dim=1)
        cls = torch.cat([self.cls_module(feature) for feature in features], dim=1)
        pred = (reg, cls)
        return pred


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


if __name__ == '__main__':
    img = torch.randn([2, 3, 600, 600])
    model = RetinaNet(num_classes=80)
    output = model(img)
    print(output[0].size())
    print(output[1].size())
