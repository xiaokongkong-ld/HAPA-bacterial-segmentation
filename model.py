import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution Block
    isnn: using the conv_block of nnUnet
    isres: residual UNet or not
    """

    def __init__(self, in_ch, out_ch, isnn=False, isbn=True, isres=False):
        super(conv_block, self).__init__()
        self.isres = isres
        self.s = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        if not isnn:
            if isbn:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True))
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                nn.BatchNorm2d(out_ch, eps=1e-5, affine=True, momentum=0.1),
                nn.LeakyReLU(inplace=True, negative_slope=1e-2),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                nn.BatchNorm2d(out_ch, eps=1e-5, affine=True, momentum=0.1),
                nn.LeakyReLU(inplace=True, negative_slope=1e-2))

    def forward(self, inputs):
        if not self.isres:
            x = self.conv(inputs)
        else:
            x = self.conv(inputs)
            skip = self.s(inputs)
            x = x + skip
        return x


class unet_encoder(nn.Module):
    """
    encoder_block
    """

    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=sz, stride=sz)
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=False)

    def forward(self, inputs):
        x = self.down(inputs)
        x = self.conv(x)
        return x


class unet_decoder(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=sz, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=False)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.n_channels = in_channels
        self.n_classes = out_channels

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv0 = conv_block(in_channels, filters[0], isnn=False, isres=False)

        self.Down1 = unet_encoder(filters[0], filters[1], sz=2)
        self.Down2 = unet_encoder(filters[1], filters[2])
        self.Down3 = unet_encoder(filters[2], filters[3])
        self.Down4 = unet_encoder(filters[3], filters[4])

        self.Up4 = unet_decoder(filters[4], filters[3])
        self.Up3 = unet_decoder(filters[3], filters[2])
        self.Up2 = unet_decoder(filters[2], filters[1])
        self.Up1 = unet_decoder(filters[1], filters[0], sz=2)

        self.Conv1 = conv_block(filters[0], filters[0])
        self.Conv2 = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print(x.shape)
        e0 = self.Conv0(x)
        # print(e0.shape)
        e1 = self.Down1(e0)
        # print(e1.shape)
        e2 = self.Down2(e1)
        # print(e2.shape)
        e3 = self.Down3(e2)
        # print(e3.shape)

        e4 = self.Down4(e3)
        # print(e4.shape)

        d4 = e4
        # print(d4.shape)
        d3 = self.Up4(d4, e3)
        # print(d3.shape)
        d2 = self.Up3(d3, e2)
        # print(d2.shape)
        d1 = self.Up2(d2, e1)
        # print(d1.shape)
        d0 = self.Up1(d1, e0)
        # print(d0.shape)

        d0 = self.Conv1(d0)
        out = self.Conv2(d0)

        return out


