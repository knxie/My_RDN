import math

import torch.nn.functional as F
from torch import nn
import torch
from collections import OrderedDict
import torch.nn.init as init
#原始生成器
# class Generator(nn.Module):
#     def __init__(self, scale_factor):
#         upsample_block_num = int(math.log(scale_factor, 2))
#
#         super(Generator, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#         self.block2 = ResidualBlock(64)
#         self.block3 = ResidualBlock(64)
#         self.block4 = ResidualBlock(64)
#         self.block5 = ResidualBlock(64)
#         self.block6 = ResidualBlock(64)
#         self.block7 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
#         block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
#         self.block8 = nn.Sequential(*block8)
#
#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         block4 = self.block4(block3)
#         block5 = self.block5(block4)
#         block6 = self.block6(block5)
#         block7 = self.block7(block6)
#         block8 = self.block8(block1 + block7)
#
#         return (F.tanh(block8) + 1) / 2


#dense unet 生成器
# class Scale(nn.Module):
#     def __init__(self, num_feature):
#         super(Scale, self).__init__()
#         self.num_feature = num_feature
#         self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
#         self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)
#
#     def forward(self, x):
#         y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
#         for i in range(self.num_feature):
#             y[:, i, :, :] = x[:, i, :, :].clone() * self.gamma[i] + self.beta[i]
#         return y


class conv_block(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0, weight_decay=1e-4):
        super(conv_block, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('conv2d1', nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1)))
        self.add_module('norm1', nn.BatchNorm2d(4 * growth_rate, eps=eps, momentum=1))

        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2d2', nn.Conv2d(4 * growth_rate, growth_rate, (3, 3), padding=(1, 1)))
        self.add_module('norm2', nn.BatchNorm2d( growth_rate, eps=eps, momentum=1))
        self.add_module('relu2', nn.ReLU(inplace=True))

        # self.drop = dropout_rate
        # self.add_module('norm1', nn.BatchNorm2d(nb_inp_fea, eps=eps, momentum=1))
        # self.add_module('scale1', Scale(nb_inp_fea))
        # self.add_module('relu1', nn.ReLU(inplace=True))
        # self.add_module('conv2d1', nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1), bias=False))
        # self.add_module('norm2', nn.BatchNorm2d(4 * growth_rate, eps=eps, momentum=1))
        # self.add_module('scale2', Scale(4 * growth_rate))
        # self.add_module('relu2', nn.ReLU(inplace=True))
        # self.add_module('conv2d2', nn.Conv2d(4 * growth_rate, growth_rate, (3, 3), padding=(1, 1), bias=False))

    def forward(self, x):
        # out = self.norm1(x)
        # out = self.scale1(out)
        # out = self.relu1(out)
        # out = self.conv2d1(out)
        #
        # if (self.drop > 0):
        #     out = F.dropout(out, p=self.drop)
        #
        # out = self.norm2(out)
        # out = self.scale2(out)
        # out = self.relu2(out)
        # out = self.conv2d2(out)
        #
        # if (self.drop > 0):
        #     out = F.dropout(out, p=self.drop)
        # print(1)
        out = self.conv2d1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        # print(2)
        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        out = self.conv2d2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        return out


# class _Transition(nn.Sequential):
#     def __init__(self, num_input, num_output, drop=0):
#         super(_Transition, self).__init__()
#         self.add_module('norm', nn.BatchNorm2d(num_input))
#         self.add_module('scale', Scale(num_input))
#         self.add_module('relu', nn.ReLU(inplace=True))
#         self.add_module('conv2d', nn.Conv2d(num_input, num_output, (1, 1), bias=False))
#         if (drop > 0):
#             self.add_module('drop', nn.Dropout(drop, inplace=True))
#         self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class dense_block(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(dense_block, self).__init__()
        for i in range(nb_layers):
            layer = conv_block(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer%d' % (i + 1), layer)

    def forward(self, x):
        # print(3)
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

#####
#denseUnet
#####
class Generator(nn.Module):
    def __init__(self, growth_rate=[8,16,32,64,128], num_init_features=32, drop_rate=0):
        super(Generator, self).__init__()

        nb_filter = num_init_features
        eps = 1.1e-5
        num_layer=4
        image_channel=1

        self.down=[]
        self.up=[]




        init = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(image_channel, nb_filter, kernel_size=3, stride=1,
                                padding=1)),
            ('norm0', nn.BatchNorm2d(nb_filter, eps=eps)),
            #('scale0', Scale(nb_filter)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        block1=dense_block(4,nb_filter,growth_rate[0], drop_rate)
        init.add_module('denseblock0',block1)
        nb_filter=nb_filter+4*growth_rate[0]
        self.down.append(init)

        for i,growth_rate_block in enumerate(growth_rate):
            if i != 0:
                down_hat=nn.Sequential()
                down_hat.add_module('pool0',nn.MaxPool2d(kernel_size=2,stride=2))
                block = dense_block(num_layer, nb_filter,growth_rate_block, drop_rate)
                nb_filter += num_layer * growth_rate_block
                down_hat.add_module('denseblock%d' % i,block)
                if i==len(growth_rate)-1:
                    # down_hat.add_module('up0',nn.Upsample(scale_factor=2))
                    down_hat.add_module('convt0',nn.ConvTranspose2d(nb_filter,int(nb_filter/2),kernel_size=2,stride=2,padding=0))
                    down_hat.add_module('normt0',nn.BatchNorm2d(int(nb_filter/2),eps=eps))
                    down_hat.add_module('relut0',nn.ReLU(inplace=True))
                    # nb_filter=int(nb_filter/2)
                self.down.append(down_hat)
        growth_rate.reverse()

        for i,growth_rate_block in enumerate(growth_rate):
            if i !=0:
                up_hat=nn.Sequential(OrderedDict([('uconv%d' % (i-1),nn.Conv2d(nb_filter,int(nb_filter/4),kernel_size=1,stride=1,padding=0)),
                                                  ('unorm%d' % (i-1),nn.BatchNorm2d(int(nb_filter/4),eps=eps)),
                                                   ('urelu%d' % (i-1),nn.ReLU(inplace=True))]))
                nb_filter=int(nb_filter/4)

                block = dense_block(num_layer, nb_filter, growth_rate_block, drop_rate)
                up_hat.add_module('udenseblock%d' % (i-1),block)
                nb_filter += num_layer * growth_rate_block
                if i==len(growth_rate)-1:
                    up_hat.add_module('res',nn.Conv2d(nb_filter,image_channel,kernel_size=1,stride=1,padding=0))
                else:
                    # up_hat.add_module('up%d' % i,nn.Upsample(scale_factor=2))
                    up_hat.add_module('convt%d' % i, nn.ConvTranspose2d(nb_filter, int(nb_filter/2), kernel_size=2, stride=2,padding=0))
                    up_hat.add_module('normt%d' % i, nn.BatchNorm2d(int(nb_filter/2), eps=eps))
                    up_hat.add_module('relut%d' % i, nn.ReLU(inplace=True))
                self.up.append(up_hat)
        self.down=nn.ModuleList(self.down)
        self.up=nn.ModuleList(self.up)
        # self.test=nn.Conv2d(3,3,kernel_size=1)
    def forward(self, x):
        down_list=[]
        down_list.append(x)
        for i,module in enumerate(self.down):
            # print(module)
            x=module(x)
            down_list.append(x)

        for i,module in enumerate(self.up):
            x=torch.cat((x,down_list[-i-2]),1)
            x=module(x)
        x=x+down_list[0]

        return x




class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self):
        super(RDN, self).__init__()
        r = 1
        G0 = 64
        kSize = 3
        n_colors=1

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (8,4,16),
        }['C']

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif r==1:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        y = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            y = self.RDBs[i](y)
            RDBs_out.append(y)

        y = self.GFF(torch.cat(RDBs_out, 1))
        y += f__1
        y=self.UPNet(y)

        return x+y
# 原始判别器
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1024, kernel_size=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1, kernel_size=1)
#         )
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         return F.sigmoid(self.net(x).view(batch_size))
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.prelu = nn.PReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)
#
#     def forward(self, x):
#         residual = self.conv1(x)
#         residual = self.bn1(residual)
#         residual = self.prelu(residual)
#         residual = self.conv2(residual)
#         residual = self.bn2(residual)
#
#         return x + residual
#
#
# class UpsampleBLock(nn.Module):
#     def __init__(self, in_channels, up_scale):
#         super(UpsampleBLock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
#         self.pixel_shuffle = nn.PixelShuffle(up_scale)
#         self.prelu = nn.PReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pixel_shuffle(x)
#         x = self.prelu(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self,encoder,decoder):
        super(Discriminator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X

class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output

# input: batch_size * nc * 128 * 128
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, isize, nc, k=100, ndf=64):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                        nn.Conv2d(cndf, k, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial-{0}-{1}-convt'.format(k, cngf), nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)