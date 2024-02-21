import torch
from torch.nn import MSELoss
from pytorch_msssim import SSIM


def PSNR(x, y, max_val=1.0):
    return 10*torch.log10(max_val/MSELoss(size_average=True)(x, y))


class PSNRWrapper():
    def __init__(self, border=4):
        self.border = border
    def __call__(self, x, y):
        x, _ = x
        x = x[:, :, self.border:-self.border, self.border:-self.border]
        y = y[:, :, self.border:-self.border, self.border:-self.border]

        return PSNR(x, y)


class SSIMWrapper():
    def __init__(self, border=4, channel=1):
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=channel, nonnegative_ssim=True).cuda()
        self.border = border

    def __call__(self, x, y):
        x, _ = x
        y = y[:, :, self.border:-self.border, self.border:-self.border]
        x = x[:, :, self.border:-self.border, self.border:-self.border]

        return self.ssim_module(x, y)
