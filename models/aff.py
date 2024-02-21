import torch
import torch.nn as nn

from models.tse.base_model import BaseModel
from models.fft_blur_deconv_pytorch import  wiener_filter, mul_fourier_conj, mul_fourier


class AAplusModel(torch.nn.Module):
    def __init__(self, H, fd_black, epsilon):
        super(AAplusModel, self).__init__()
        self.A = H
        self.Aplus = wiener_filter(H, epsilon=epsilon)
        self.fd_black = fd_black

    def rfft(self, x):
        x_mean = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        x = x-x_mean
        x = self.fd_black.pad(x)
        Fx = torch.rfft(x, 2, onesided=False)

        return Fx, x_mean

    def irfft(self, Fx, x_mean):
        x = torch.irfft(Fx, 2, onesided=False, signal_sizes=Fx.shape[-3:-1])
        x = self.fd_black.unpad(x)
        x = x+x_mean

        return x

    def A_only(self, x):
        Fx, x_mean = self.rfft(x)
        Fx = mul_fourier(Fx, self.A)
        x = self.irfft(Fx, x_mean)

        return x

    def Aplus_only(self, x):
        Fx, x_mean = self.rfft(x)
        Fx = mul_fourier_conj(Fx, self.Aplus)
        x = self.irfft(Fx, x_mean)

        return x

    def forward(self, x):
        Fx, x_mean = self.rfft(x)
        Fx = mul_fourier(Fx, self.A)
        Fx = mul_fourier_conj(Fx, self.Aplus)
        x = self.irfft(Fx, x_mean)

        return x


class DynamicFilter(nn.Module):
    def __init__(self, kernel_size=5):
        super(DynamicFilter, self).__init__()
        self.add_module('unfold', nn.Unfold(kernel_size=kernel_size, dilation=1,
                                            padding=kernel_size//2, stride=1))

    def forward(self, x, filters):
        batch, channels, height, width = x.size()

        v_out = self.unfold(x).view(batch, channels, -1, height, width) #[b, c, k, h, w]
        k_out = filters.view(batch, 1, -1, height, width) #[b, k, h, w]
        out = (v_out*k_out).sum(dim=2) #[b, c, h, w]

        return out


class DenseResidualBlock(nn.Module):
    def __init__(self, inplanes, outplanes, k, pad, stride=1, bias=True,
                 groups=1, dilation=1):
        super().__init__()
        self.add_module('cv1', nn.Sequential(
            nn.Conv2d(inplanes, outplanes, (k, k), stride=stride,
                      padding=pad, bias=bias, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        self.add_module('cv2', nn.Sequential(
            nn.Conv2d(inplanes+outplanes, outplanes, (k, k), stride=stride,
                      padding=pad, bias=bias, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        self.add_module('cv3', nn.Sequential(
            nn.Conv2d(inplanes+outplanes*2, outplanes, (k, k), stride=stride,
                      padding=pad, bias=bias, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        self.add_module('fuse', nn.Sequential(
            nn.Conv2d(inplanes+outplanes*3, inplanes, (1, 1), stride=stride,
                      padding=0, bias=bias)
        ))

    def forward(self, x):
        res = x
        x = torch.cat([self.cv1(x), x], dim=1)
        x = torch.cat([self.cv2(x), x], dim=1)
        x = torch.cat([self.cv3(x), x], dim=1)
        x = self.fuse(x)

        return x+res


class CustomPixelShuffle(nn.Module):
    def __init__(self, factor):
        super(CustomPixelShuffle, self).__init__()
        self.factor = factor

    def forward(self, x):
        out_height, out_width = self.factor*x.size(2), self.factor*x.size(3)
        x = x.view(x.size(0), -1, self.factor, self.factor, x.size(2), x.size(3))
        x = x.permute(0, 1, 4, 3, 5, 2).contiguous()
        x = x.view(x.size(0), -1, out_height, out_width)

        return x


class CustomDePixelShuffle(nn.Module):
    def __init__(self, factor):
        super(CustomDePixelShuffle, self).__init__()
        self.factor = factor

    def forward(self, x):
        out_height, out_width = int(x.size(2)//self.factor), int(x.size(3)//self.factor)
        x = x.view(x.size(0), x.size(1), out_height, self.factor, out_width, self.factor)
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
        x = x.view(x.size(0), -1, out_height, out_width)

        return x


class DDNet(BaseModel):
    def __init__(self, channels_c=1, n_features=64, nb=10,
                 res_block=DenseResidualBlock, df_size=5):
        super().__init__()
        self.add_module('in_cv', nn.Sequential(
                        CustomDePixelShuffle(2),
                        nn.Conv2d(channels_c*2*4, n_features, 3, 1, 1, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        CustomDePixelShuffle(2),
                        nn.Conv2d(n_features*4, n_features, 3, 1, 1, bias=True),
                        nn.LeakyReLU(0.2, inplace=True)
                    ))

        res_blocks = []

        for i in range(nb):
            layer = res_block(n_features, n_features, 3, pad=1, stride=1,
                              bias=True, dilation=1)
            res_blocks.append(layer)

        self.add_module('res_blocks', nn.Sequential(*res_blocks))

        self.add_module('fuse_cv', nn.Sequential(
                        nn.Conv2d(n_features, n_features, 3, 1, 1, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                    ))
        self.add_module('upsample', nn.Sequential(
                        nn.Conv2d(n_features, n_features*4, 3, 1, 1, bias=True),
                        CustomPixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(n_features, n_features*4, 3, 1, 1, bias=True),
                        CustomPixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                        ))
        self.add_module('convout_im', nn.Sequential(
            nn.Conv2d(n_features, n_features, (3, 3), 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features, channels_c, (1, 1), 1, padding=0)
        ))
        self.add_module('convout_df', nn.Sequential(
            nn.Conv2d(n_features, n_features, (3, 3), 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features, df_size**2, (1, 1), 1, padding=0)
        ))

    def forward(self, x, amasx3):
        x = torch.cat([x, amasx3], dim=1)
        in_cv = self.in_cv(x)
        x = self.res_blocks(in_cv)
        x = self.fuse_cv(x+in_cv)
        x = self.upsample(x)
        h_out = self.convout_df(x)
        deblur_out = self.convout_im(x)

        return deblur_out, h_out


class AffWrapper(BaseModel):
    def __init__(self, Generator, df_size=5):
        super().__init__()
        self.Generator = Generator
        self.add_module('df', DynamicFilter(df_size))

    def forward(self, x):
        amortised_model, Amas_x, x = x

        fx, h = self.Generator(x, Amas_x)
        auxfx = amortised_model(fx)
        out = fx - auxfx + Amas_x
        out = self.df(out, h)

        return out, h


class ColorWrapper(BaseModel):
    def __init__(self, Generator):
        super().__init__()
        self.Generator = Generator

    def forward(self, x):
        Amas_x, x = x
        out, h = self.Generator(x, Amas_x)

        return out, h
