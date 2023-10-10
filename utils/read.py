from os import listdir
from os.path import basename, isdir, join, splitext

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import scipy.io

from functools import partial

import re


def to_rgb1(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def get_files_path_from_path(path, pattern, ext):
    files = []
    ext_padded = ['/' + s + '/' for s in ext]

    for f in listdir(path):
        f_c = join(path, f)

        if isdir(f_c):
            files = files + get_files_path_from_path(f_c, pattern, ext)
        else:
            f_n, f_e = splitext(basename(f_c))

            test_string = '/' + f_e.lower() + '/'

            if (test_string in ext_padded):
                if pattern is None:
                    files.append(f_c)
                elif re.match(pattern, f_n):
                    files.append(f_c)

    return files


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)


def corr_fourier(a, b):
    """
    Given a and b two 5-dimensional tensors
    with the last dimension being the real and imaginary part,
    returns the convolution of a and b.
    """
    op = partial(torch.einsum, "bcij,bpij->bcij")
    return torch.stack([
        op(a[..., 0], b[..., 0]) + op(a[..., 1], b[..., 1]),
        -op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ],
                       dim=-1)


def rgb2ycbcr(img, only_y=False):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    img = img[::-1].transpose((1, 2, 0))
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        rlt = np.expand_dims(rlt, axis=0)
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        rlt = rlt.transpose((2, 0, 1))
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


class RestorationsDataset(Dataset):
    def __init__(self, root_dir, extension, im_size, regularizer, transform=None):
        from models.fft_blur_deconv_pytorch import psf2otf

        """
        Args:
            root_dir (string): Directory with all the tripletes (gt, degraded, restored).
            psfs_file (string): .mat file with all the psfs.
            extension (string): File extensions to consider. Currently only hdf5 and mat are supported.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filelist = get_files_path_from_path(root_dir, '^[0-9_]*$', extension)
        self.root_dir = root_dir
        self.extension = extension
        self.transform = transform

        self.im_size = im_size
        self.regularizer = regularizer

    def __len__(self):
        return len(self.filelist)

    def __filename__(self,idx):
        return self.filelist[idx]

    def __getitem__(self, idx):
        file = self.filelist[idx]
        thename, ext = splitext(basename(file))
        file_psf = splitext(file)[0]+'_H'+ext
        originalimageidx = thename.split('_')[0]
        psfnumber = int(thename.split('_')[1])
        ext = splitext(file)[1]

        if ext.lower() == '.mat':
            mat = scipy.io.loadmat(file)
            x = mat['X'][:] #Real
            y = mat['Y'][:] #Degraded
            H = mat['H'][:].astype(np.float32)/32768.0
        elif ext.lower() == '.npy':
            aux = np.load(file)
            x = rgb2ycbcr(aux[:3], only_y=True) #Real
            y = rgb2ycbcr(aux[3:6], only_y=True) #Degraded
            H = np.load(file_psf)

        image = np.concatenate([x, y], axis=0)

        if self.transform:
            image = self.transform(image)

        x = np.ascontiguousarray(image[0:1])
        y = np.ascontiguousarray(image[1:2])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        H =  torch.from_numpy(H)

        return [x, y, H]


class RestorationsColorOnlyDataset(Dataset):
    def __init__(self, root_dir, extension, im_size, regularizer, transform=None):
        from models.fft_blur_deconv_pytorch import psf2otf

        """
        Args:
            root_dir (string): Directory with all the tripletes (gt, degraded, restored).
            psfs_file (string): .mat file with all the psfs.
            extension (string): File extensions to consider. Currently only hdf5 and mat are supported.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filelist = get_files_path_from_path(root_dir, '^[0-9_]*$', extension)
        self.root_dir = root_dir
        self.extension = extension
        self.transform = transform

        self.im_size = im_size
        self.regularizer = regularizer

    def __len__(self):
        return len(self.filelist)

    def __filename__(self,idx):
        return self.filelist[idx]

    def __getitem__(self, idx):
        file = self.filelist[idx]
        thename, ext = splitext(basename(file))
        file_psf = splitext(file)[0]+'_H'+ext
        originalimageidx = thename.split('_')[0]
        psfnumber = int(thename.split('_')[1])
        #True_H = self.psfs_TH[psfnumber]
        ext = splitext(file)[1]

        if ext.lower() == '.mat':
            mat = scipy.io.loadmat(file)
            x = mat['X'][:] #Real
            y = mat['Y'][:] #Degraded
            H = mat['H'][:].astype(np.float32)/32768.0
        elif ext.lower() == '.npy':
            aux = np.load(file)
            x = rgb2ycbcr(aux[:3], only_y=False)[1:] #Real
            y = rgb2ycbcr(aux[3:6], only_y=False)[1:] #Degraded
            H = np.load(file_psf)

        image = np.concatenate([x, y], axis=0)

        if self.transform:
            image = self.transform(image)

        x = np.ascontiguousarray(image[0:2])
        y = np.ascontiguousarray(image[2:4])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        H =  torch.from_numpy(H)

        return [x, y, H]


class RestorationsEvalColorDataset(Dataset):
    def __init__(self, im_dir, psf_dir, extension):
        """
        Args:
            im_dir (string): Directory with all the images.
            psf_dir (string): Directory with all the psfs in .mat file format.
            extension (string): File extensions to consider for image files.
        """
        #self.filelist = get_files_path_from_path(im_dir, '^[0-9_]*$', extension)
        self.filelist = get_files_path_from_path(im_dir, None, extension)
        self.im_dir = im_dir
        self.psf_dir = psf_dir
        self.extension = extension

    def __len__(self):
        return len(self.filelist)

    def __filename__(self, idx):
        return self.filelist[idx]

    def __getitem__(self, idx):
        file = self.filelist[idx]
        thename, ext = splitext(basename(file))
        file_psf = join(self.psf_dir, thename+'_psf.npy')
        y = rgb2ycbcr(cv2.imread(file, 1).transpose((2,0,1))[::-1])
        y = torch.from_numpy(y)
        y_color = y[1:].contiguous()
        y = y[:1].contiguous()
        #psf = scipy.io.loadmat(file_psf)['kernel']
        psf = np.load(file_psf)
        psf = torch.from_numpy(psf.astype(np.float64)).squeeze().unsqueeze(dim=0)

        return [y, y_color, psf]