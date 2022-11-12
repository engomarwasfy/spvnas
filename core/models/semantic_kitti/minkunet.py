import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

__all__ = ['MinkUNet']


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
            self.downsample = nn.Sequential()
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


class MinkUNet(nn.Module):

    def __init__(self,input_kernel_maps,number_of_encoding_layers,cs,decoder,number_of_classes,**kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        #cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.number_of_encoding_layers = number_of_encoding_layers
        self.decoder = decoder
        self.cs = cs

        self.stem = nn.Sequential(
            spnn.Conv3d(input_kernel_maps, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))
        self.encoders =nn.Sequential()
        self.decoders = nn.Sequential()
        for i in range(0,number_of_encoding_layers-1):
            self.encoders.append(nn.Sequential(
                BasicConvolutionBlock(cs[i], cs[i], ks=2, stride=2, dilation=1),
                ResidualBlock(cs[i], cs[i+1], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[i+1], cs[i+1], ks=3, stride=1, dilation=1)))
            j=i+number_of_encoding_layers
            self.decoders.append(nn.ModuleList([
            BasicDeconvolutionBlock(cs[j], cs[j+1], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[j+1] + cs[number_of_encoding_layers-i-1], cs[j+1], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[j+1], cs[j+1], ks=3, stride=1, dilation=1),
            )
        ]))
        self.classifier = nn.Sequential(nn.Linear(cs[-1], number_of_classes))
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs=[]
        ys=[]
        xs.append(self.stem(x))
        for i in range(0,self.number_of_encoding_layers-1):
            xs.append(self.encoders[i](xs[i]))
        ys.append(xs[-1])
        for i in range(0,self.number_of_encoding_layers-1):
            y=self.decoders[i][0](ys[i])
            y=torchsparse.cat((y,xs[self.number_of_encoding_layers-i-2]))
            y=self.decoders[i][1](y)
            ys.append(y)
        out = ys[-1]

        return out

class U2NET(nn.Module):

    def __init__(self,number_of_encoding_layers,cs,**kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        #cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.cs = cs
        self.number_of_encoding_layers = number_of_encoding_layers
        number_of_classes =kwargs['num_classes']

        self.downsize = nn.Sequential(
            spnn.Conv3d(2*cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))
        self.encoders =nn.Sequential()
        self.decoders = nn.Sequential()
        length=len(cs)
        for i in range(0,number_of_encoding_layers-1):
            self.encoders.append(MinkUNet(cs[i],number_of_encoding_layers=number_of_encoding_layers-i,decoder=False,cs=cs[i:length-1-i],number_of_classes=number_of_classes))
            self.decoders.append(MinkUNet(cs[i],number_of_encoding_layers=number_of_encoding_layers-i,decoder=True,cs=cs[i:length-1-i],number_of_classes=number_of_classes))
        self.classifier = nn.Sequential(nn.Linear(cs[-1]*4, kwargs['num_classes']))
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs=[]
        ys=[]
        xs.append(self.stem(x))
        for i in range(0,self.number_of_encoding_layers-1):
            xs.append(self.encoders[i].forward(xs[i]))
        ys.append(xs[-1])
        y=xs[-1]
        y = self.decoders[0].forward(y)
        y = torchsparse.cat((y, xs[self.number_of_encoding_layers - 2]))
        y = self.downsize(y)
        ys.append(y)
        for i in range(1,self.number_of_encoding_layers-1):
            y=self.decoders[i].forward(ys[i-1])
            y=torchsparse.cat((y,xs[self.number_of_encoding_layers-i-2]))
            y=self.downsize(y)
            ys.append(y)
        z=torchsparse.cat((ys[-1],ys[-2],ys[-3],ys[-4]))
        out = self.classifier(z.F)

        return out

