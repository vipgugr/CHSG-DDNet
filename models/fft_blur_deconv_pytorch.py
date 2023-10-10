import numpy as np
import torch


def mul_fourier(t1, t2):
  real1, imag1 = torch.chunk(t1, 2, dim=-1)
  real2, imag2 = torch.chunk(t2, 2, dim=-1)

  return torch.cat([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)


def mul_fourier_conj(t1, t2):
  real1, imag1 = torch.chunk(t1, 2, dim=-1)
  real2, imag2 = torch.chunk(t2, 2, dim=-1)

  return torch.cat([real1 * real2 + imag1 * imag2, -real1 * imag2 + imag1 * real2], dim = -1)


def div_fourier(t1, t2, epsilon):
  real1, imag1 = torch.chunk(t1, 2, dim=-1)
  real2, imag2 = torch.chunk(t2, 2, dim=-1)
  real2 = real2 + epsilon

  return torch.cat([real1/real2 - imag1*imag2, real1 * imag2 + imag1/real2], dim = -1)


def div_fourier_conj(t1, t2, epsilon):
  real1, imag1 = torch.chunk(t1, 2, dim=-1)
  real2, imag2 = torch.chunk(t2, 2, dim=-1)
  real2 = real2 + epsilon

  return torch.cat([real1/real2 + imag1*imag2, -real1 * imag2 + imag1/real2], dim = -1)


def real_if_close(t, tol):
    if tol > 1:
        from numpy.core import getlimits
        type = t.type()

        if type == 'torch.FloatTensor' or type == 'torch.cuda.FloatTensor':
            type = 'float'
        elif type=='torch.DoubleTensor' or type=='torch.cuda.DoubleTensor':
            type = 'double'

        f = getlimits.finfo(type)
        tol = f.eps * tol

    if torch.all(torch.abs(t[..., 1]) < tol):
        t = torch.real(t)

    return t

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape[-2:], dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = torch.zeros(tuple(shape), dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[..., idx + offx, idy + offy] = image

    return pad_img

'''
def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape[-2:], dtype=int)
    pad_size = shape-imshape
    pad_img = torch.nn.functional.pad(image, (0, pad_size[1], 0, pad_size[0]),
                mode='constant', value=0)

    return pad_img
'''

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if torch.all(psf == 0):
        return torch.zeros_like(psf)

    inshape = psf.shape[-2:]
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    axis_roll = [-int(axis_size / 2) for axis, axis_size in enumerate(inshape)]
    psf = torch.roll(psf, axis_roll, dims=[i for i in range(len(inshape))])

    # Compute the OTF
    otf = torch.rfft(psf, 2, onesided=False)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(np.product(psf.shape) * np.log2(psf.shape))
    otf = real_if_close(otf, tol=n_ops)

    return otf


def conj(t):
    real,imag = torch.chunk(t, 2, dim=-1)
    return torch.cat([real, -imag], dim=-1)


def wiener_filter(FA, epsilon=0.01):
    #Numerator
    numerator = FA

    #Denominator
    denominator = mul_fourier_conj(FA, FA)

    #We divive term by term
    FAplus = div_fourier_conj(numerator, denominator, epsilon)

    return FAplus


def fft_blur(x, psf):
    #Calculate A
    FA = psf2otf(psf, x.shape[-2:])

    #F(x)
    or_shape = x.shape[-2:]
    x_mean = x.mean()
    x = torch.rfft(x-x_mean, 2, onesided=False)

    #Apply A
    y = mul_fourier(x, FA)

    #iF(y)
    y = torch.irfft(y, 2, onesided=False, signal_sizes=or_shape) + x_mean

    return y


def deconv_weiner(y, psf, epsilon):
    #Calculate A and Aplus
    FA = psf2otf(psf, y.shape[-2:])
    FAplus = wiener_filter(FA, epsilon=epsilon)

    #F(y)
    or_shape = y.shape[-2:]
    y_mean = y.mean()
    y = torch.rfft(y-y_mean, 2, onesided=False)

    #Apply Aplus
    x = mul_fourier_conj(y, FAplus)

    #iF(x)
    x = torch.irfft(x, 2, onesided=False, signal_sizes=or_shape) + y_mean

    return x
