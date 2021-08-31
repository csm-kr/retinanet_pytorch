import math
import torch
import torch.nn as nn
from torchvision.models import resnet50


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


class FPNExtractor(nn.Module):
    def __init__(self, c3, c4, c5, inner_channel=256, bias=True):
        super(FPNExtractor, self).__init__()
        self.c3_latent = nn.Conv2d(c3, inner_channel, 1, 1, 0, bias=bias)
        self.c4_latent = nn.Conv2d(c4, inner_channel, 1, 1, 0, bias=bias)
        self.c5_latent = nn.Conv2d(c5, inner_channel, 1, 1, 0, bias=bias)
        self.c5_to_c6 = nn.Conv2d(c5, inner_channel, 3, 2, 1, bias=bias)
        self.c6_to_c7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(inner_channel, inner_channel, 3, 2, 1, bias=bias)
        )

    def forward(self, xs):
        c3, c4, c5 = xs
        f3 = self.c3_latent(c3)
        f4 = self.c4_latent(c4)
        f5 = self.c5_latent(c5)
        f6 = self.c5_to_c6(c5)
        f7 = self.c6_to_c7(f6)
        return [f3, f4, f5, f6, f7]


class FPN(nn.Module):
    def __init__(self, c3, c4, c5, out_channel, bias=True):
        super(FPN, self).__init__()
        self.fpn_extractor = FPNExtractor(c3, c4, c5, out_channel, bias)
        self.p3_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        self.p4_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        self.p5_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)

    def forward(self, xs):
        f3, f4, f5, f6, f7 = self.fpn_extractor(xs)
        p5 = self.p5_out(f5)
        f4 = f4 + nn.UpsamplingBilinear2d(size=(f4.shape[2:]))(f5)
        p4 = self.p4_out(f4)
        f3 = f3 + nn.UpsamplingBilinear2d(size=(f3.shape[2:]))(f4)
        p3 = self.p3_out(f3)
        return [p3, p4, p5, f6, f7]


class RetinaNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=80):
        super(RetinaNet, self).__init__()

        self.backbone = Resnet50(pretrained=pretrained)
        self.neck = FPN(512, 1024, 2048, out_channel=256)
        self.num_classes = num_classes
        self.cls_head = ClsHead(self.num_classes)
        self.reg_head = RegHead()

        self.initialize_subsets()
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_subsets(self):

        for i, c in enumerate(self.cls_head.features.children()):
            if isinstance(c, nn.Conv2d):
                if i == 12:  # final layer
                    nn.init.constant_(c.bias, -math.log((1 - 0.01) / 0.01))
                    nn.init.normal_(c.weight, std=0.01)
                else:
                    nn.init.normal_(c.weight, std=0.01)
                    nn.init.constant_(c.bias, 0)
        for c in self.reg_head.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    # https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/model.py
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        xs = self.backbone(x)
        features = self.neck(xs)  # [p3, p4, p5, p6, p7 ]
        cls = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        reg = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        return cls, reg


class ClsHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
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


class RegHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.GroupNorm(32, 256),
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
    model = RetinaNet(pretrained=False, num_classes=80)
    output = model(img)
    print(output[0].size())
    print(output[1].size())