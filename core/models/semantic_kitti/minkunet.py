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

    def __init__(self,input_kernel_maps,output_kernel_maps,number_of_encoding_layers,cs,decoder,number_of_classes,**kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        #cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
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
        self.stemOut = nn.Sequential(
            spnn.Conv3d(cs[-1],output_kernel_maps , kernel_size=1, stride=1),
            spnn.BatchNorm(output_kernel_maps), spnn.ReLU(True)
        )
        self.encoders =nn.Sequential()
        self.decoders = nn.Sequential()
        for i in range(0,number_of_encoding_layers):
            self.encoders.append(nn.Sequential(
                BasicConvolutionBlock(cs[i], cs[i], ks=2, stride=2, dilation=1),
                ResidualBlock(cs[i], cs[i+1], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[i+1], cs[i+1], ks=3, stride=1, dilation=1)))

            self.decoders.append(nn.ModuleList([
                BasicDeconvolutionBlock(cs[number_of_encoding_layers+i], cs[number_of_encoding_layers+i+1], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[number_of_encoding_layers+i+1] + cs[number_of_encoding_layers-i-1], cs[number_of_encoding_layers+i+1], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[number_of_encoding_layers+i+1], cs[number_of_encoding_layers+i+1], ks=3, stride=1, dilation=1),
                )
                ]))
        #self.classifier = nn.Sequential(nn.Linear(cs[-1], number_of_classes))
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
        xs.append(self.stemIn(x))
        for i in range(0,self.number_of_encoding_layers):
            xs.append(self.encoders[i](xs[i]))
        ys.append(xs[-1])
        for i in range(0,self.number_of_encoding_layers):
            y=self.decoders[i][0](ys[i])
            y=torchsparse.cat((y,xs[self.number_of_encoding_layers-i-1]))
            y=self.decoders[i][1](y)
            ys.append(y)
        out = self.stemOut(ys[-1])

        return out

class U2NET(nn.Module):

    def __init__(self,u2Interleavinglayers,number_of_encoding_layers,cs,**kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        #cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.cs = cs
        self.number_of_encoding_layers = number_of_encoding_layers
        number_of_classes =kwargs['num_classes']

        self.upsample = nn.Sequential(spnn.Conv3d(u2Interleavinglayers,u2Interleavinglayers , kernel_size=1, stride=1, transposed=True),
            spnn.BatchNorm(u2Interleavinglayers), spnn.ReLU(True))
        self.stem = nn.Sequential(
            spnn.Conv3d(4, u2Interleavinglayers, kernel_size=3, stride=1),
            spnn.BatchNorm(u2Interleavinglayers), spnn.ReLU(True),
            spnn.Conv3d(u2Interleavinglayers, u2Interleavinglayers, kernel_size=3, stride=1),
            spnn.BatchNorm(u2Interleavinglayers), spnn.ReLU(True))
        self.encoders =nn.Sequential()
        self.decoders = nn.Sequential()
        length=len(cs)
        decoder_Weight_maps_count=0
        for i in range(0,number_of_encoding_layers):
            self.encoders.append(MinkUNet(u2Interleavinglayers,u2Interleavinglayers,number_of_encoding_layers=number_of_encoding_layers-i,decoder=False,cs=cs[i:length-i],number_of_classes=number_of_classes))
            self.decoders.append(MinkUNet(u2Interleavinglayers*2,u2Interleavinglayers,number_of_encoding_layers=number_of_encoding_layers-i,decoder=True,cs=cs[i:length-i],number_of_classes=number_of_classes))
            decoder_Weight_maps_count+=u2Interleavinglayers
        self.classifier = nn.Sequential(nn.Linear(decoder_Weight_maps_count, kwargs['num_classes']))
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
        for i in range(0,self.number_of_encoding_layers):
            xs.append(self.encoders[i].forward(xs[i]))
        ys.append(xs[-1])
        y=xs[-1]
        y=torchsparse.cat((y,y))
        y = self.decoders[0].forward(y)
        ys.append(y)
        for i in range(1,self.number_of_encoding_layers-1):
            y=torchsparse.cat((ys[i],xs[-i]))
            y = self.decoders[i].forward(y)
            ys.append(y)
        z=torchsparse.cat((ys[-1],ys[-2],ys[-3],ys[-4]))
        out = self.classifier(z.F)

        return out

