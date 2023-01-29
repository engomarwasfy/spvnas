import torchsparse
import torchsparse.nn as spnn
from torch import nn
from torchsparse import PointTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN']


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):

    def __init__(self, input_kernel_maps, output_kernel_maps, number_of_encoding_layers, cs, decoder, number_of_classes,
                 **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        # cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.number_of_encoding_layers = number_of_encoding_layers
        self.decoder = decoder
        self.cs = cs

        self.stemIn = nn.Sequential(
            spnn.Conv3d(input_kernel_maps, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))
        self.encoders = nn.Sequential()
        self.decoders = nn.Sequential()
        for i in range(0, number_of_encoding_layers):
            self.encoders.append(nn.Sequential(
                BasicConvolutionBlock(cs[i], cs[i], ks=2, stride=2, dilation=1),
                ResidualBlock(cs[i], cs[i + 1], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[i + 1], cs[i + 1], ks=3, stride=1, dilation=1)))

            self.decoders.append(nn.ModuleList([
                BasicDeconvolutionBlock(cs[number_of_encoding_layers + i], cs[number_of_encoding_layers + i + 1], ks=2,
                                        stride=2),
                nn.Sequential(
                    ResidualBlock(cs[number_of_encoding_layers + i + 1] + cs[number_of_encoding_layers - i - 1],
                                  cs[number_of_encoding_layers + i + 1], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[number_of_encoding_layers + i + 1], cs[number_of_encoding_layers + i + 1], ks=3,
                                  stride=1, dilation=1),
                )
            ]))
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = []
        ys = []
        zs = []
        z = PointTensor(x.F, x.C.float())
        zs.append(z)
        x0 = initial_voxelize(z, self.pres, self.vres)
        xs.append(self.stemIn(x0))
        z0 = voxel_to_point(xs[0], zs[0], nearest=False)
        x1 = point_to_voxel(x0, z0)

        # xs.append(x)
        for i in range(0, self.number_of_encoding_layers):
            zi = voxel_to_point(xs[i+1], zs[i+1], nearest=False)
            xs.append(self.encoders[i](xs[i]))
        ys.append(xs[-1])
        for i in range(0, self.number_of_encoding_layers):
            z1 = voxel_to_point(xs[-i-1], z0)
            z1.F = z1.F + self.point_transforms[0](z0.F)
            y1 = point_to_voxel(xs[-i-1], z1)
            y1.F = self.dropout(y1.F)
            y = self.decoders[i][0](ys[i])
            y = torchsparse.cat((y, xs[self.number_of_encoding_layers - i - 1]))
            y = self.decoders[i][1](y)
            ys.append(y)
        out = (ys[-1])

        return out
