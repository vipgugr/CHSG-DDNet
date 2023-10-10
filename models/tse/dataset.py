# -*- coding: utf-8 -*-
from random import choice, uniform

import numpy as np
import torch
from torch.utils.data import Dataset
from .transformations import trans_none


class NotListException(ValueError):
    pass


class BadLenthException(ValueError):
    pass


class TensorTransformsDataset(Dataset):
    def __init__(self, data, labels, trans=trans_none):
        super(TensorTransformsDataset, self).__init__()
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
        self.trans = trans

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        data, labels = self.trans(data, labels)
        data = torch.from_numpy(np.ascontiguousarray(data))
        labels = torch.from_numpy(np.ascontiguousarray(labels))

        return data, labels

    def __len__(self):
        return len(self.data)


class MultipleInputsOutputsDataset(Dataset):
    def __init__(self, data, labels, transforms=trans_none, weight=None):
        super(MultipleInputsOutputsDataset, self).__init__()
        self.set_data(data, labels)
        self.transforms = transforms
        self.weights = weight

    def set_data(self, data, labels):
        if not isinstance(data, list) or  not isinstance(labels, list):
            raise NotListException

        if len(data) == 0 or len(labels) == 0:
            raise BadLenthException

        ini_lenth = len(data[0])

        for d in data[1:]:
            if ini_lenth != len(d):
                raise BadLenthException

        for l in labels:
            if ini_lenth != len(l):
                raise BadLenthException

        self.data = data
        self.labels = labels
        self.weights = torch.ones(data[0].size(0))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        data = []
        labels = []

        for x, y in zip(self.data, self.labels):
            x, y = self.transforms(x[idx], y[idx])
            x = torch.from_numpy(np.ascontiguousarray(x))
            y = torch.from_numpy(np.ascontiguousarray(y))
            data.append(x)
            labels.append(y)

        if self.weights is None:
            return data, labels
        else:
            return data, labels, self.weights[idx]



class MultipleInputsOutputsDatasetTClass(MultipleInputsOutputsDataset):
    def __getitem__(self, idx):
        data = []
        labels = []

        for x, y in zip(self.data, self.labels):
            x, y = self.transforms[y[idx]](x[idx], y[idx])
            x = torch.from_numpy(np.ascontiguousarray(x))
            y = torch.from_numpy(np.ascontiguousarray(y))
            data.append(x)
            labels.append(y)

        return data, labels


def none_op(x):
    return x


def apply_blur(x, g):
    from scipy.ndimage import convolve

    x = x.numpy()
    g = g[0, :, :].numpy()
    y = np.zeros_like(x)

    y[0, :, :] = convolve(x[0, :, :], g, mode='nearest')
    y[1, :, :] = convolve(x[1, :, :], g, mode='nearest')
    y[2, :, :] = convolve(x[2, :, :], g, mode='nearest')

    return torch.from_numpy(y)


class MultipleInputsOutputsDatasetBIDSR(Dataset):
    def __init__(self, data, blur_params, scale, transforms=trans_none):
        super(MultipleInputsOutputsDatasetBIDSR, self).__init__()

        if not isinstance(blur_params, list):
            raise NotListException

        if len(data) == 0 or len(blur_params) == 0:
            raise BadLenthException

        self.data = data
        self.blurs, self.blurs_pca, self.max_noise, self.blurs_w = blur_params
        self.transforms = transforms
        self.scale = scale
        self.idx_blur = np.arange(self.blurs.size(0))
        np.random.shuffle(self.idx_blur)
        self.i_blur = 0

    def __len__(self):
        return len(self.data)

    def get_random_kernels(self):
        gauss_b_idx = self.idx_blur[self.i_blur]#choice(self.idx_blur)
        self.i_blur += 1

        if (self.i_blur == len(self.idx_blur)):
                np.random.shuffle(self.idx_blur)
                self.i_blur = 0

        gauss_b = self.blurs[gauss_b_idx]
        params = torch.zeros(self.blurs_pca.size(1)+1)
        params[:-1] = self.blurs_pca[gauss_b_idx]
        w = self.blurs_w[gauss_b_idx]

        return params, gauss_b, w

    def get_noise(self, data, params):
        noise_var = uniform(0.0, self.max_noise)
        params[-1] = noise_var
        shape = [data.size(0), int(data.size(1)//self.scale), int(data.size(2)//self.scale)]
        noise = torch.zeros(shape).normal_(mean=0.0, std=noise_var)

        return noise

    def __getitem__(self, idx):
        x = self.data[idx]
        x, x = self.transforms(x, x)
        x = torch.from_numpy(np.ascontiguousarray(x))
        blur_params, blur_kernels, w = self.get_random_kernels()
        #noise_var = self.get_noise(x, blur_params)
        #data = [data.float()/255.0, blur_kernels, noise_var, blur_params]
        #data = [x, blur_kernels, noise_var, blur_params]
        data = [x, blur_kernels, self.max_noise, blur_params]

        return data, [blur_params, x, w]


'''
class OemMultipleInputsOutputsDatasetBIDSR(Dataset):
    def __init__(self, data, blur_params, blur_params_an, max_noise, scale,
                 transforms=trans_none, prob_an=0.5):
        super(OemMultipleInputsOutputsDatasetBIDSR, self).__init__()

        if not isinstance(blur_params, list):
            raise NotListException

        if not isinstance(blur_params_an, list):
            raise NotListException

        if len(data) == 0 or len(blur_params) == 0 or len(blur_params_an) == 0:
            raise BadLenthException

        self.data = data
        self.max_noise = max_noise
        self.blurs, self.blurs_pca, self.blurs_w = blur_params
        self.blurs_an, self.blurs_pca_an, self.blurs_w_an = blur_params_an
        self.transforms = transforms
        self.scale = scale
        self.idx_blur = np.arange(self.blurs.size(0))
        self.idx_blur_an = np.arange(self.blurs_an.size(0))
        np.random.shuffle(self.idx_blur)
        np.random.shuffle(self.idx_blur_an)
        self.i_blur = 0
        self.i_blur_an = 0
        self.prob_an = prob_an

    def __len__(self):
        return len(self.data)

    def get_kernel(self):
        gauss_b_idx = self.idx_blur[self.i_blur]#choice(self.idx_blur)
        self.i_blur += 1

        if (self.i_blur == len(self.idx_blur)):
                np.random.shuffle(self.idx_blur)
                self.i_blur = 0

        gauss_b = self.blurs[gauss_b_idx]
        params = self.blurs_pca[gauss_b_idx]
        w = self.blurs_w[gauss_b_idx]

        return params, gauss_b, w

    def get_kernel_an(self):
        gauss_b_idx = self.idx_blur_an[self.i_blur_an]#choice(self.idx_blur)
        self.i_blur_an += 1

        if (self.i_blur_an == len(self.idx_blur_an)):
                np.random.shuffle(self.idx_blur_an)
                self.i_blur_an = 0

        gauss_b = self.blurs_an[gauss_b_idx]
        params = self.blurs_pca_an[gauss_b_idx]
        w = self.blurs_w_an[gauss_b_idx]

        return params, gauss_b, w

    def get_random_kernels(self):
        if np.random.rand() >= self.prob_an:
            params_b, gauss_b, w = self.get_kernel()
        else:
            params_b, gauss_b, w = self.get_kernel_an()

        params = torch.zeros(params_b.size(0)+1)
        params[:-1] = params_b

        return params, gauss_b, w

    def get_noise(self, data, params):
        noise_var = uniform(0.0, self.max_noise)
        params[-1] = noise_var
        shape = [data.size(0), int(data.size(1)//self.scale), int(data.size(2)//self.scale)]
        noise = torch.zeros(shape).normal_(mean=0.0, std=noise_var)

        return noise

    def __getitem__(self, idx):
        x = self.data[idx]
        x, x = self.transforms(x, x)
        x = torch.from_numpy(np.ascontiguousarray(x))
        blur_params, blur_kernels, w = self.get_random_kernels()
        #noise_var = self.get_noise(x, blur_params)
        #data = [data.float()/255.0, blur_kernels, noise_var, blur_params]
        #data = [x, blur_kernels, noise_var, blur_params]
        data = [x, blur_kernels, self.max_noise, blur_params]

        return data, [blur_params, x, w]
'''


class OemMultipleInputsOutputsDatasetBIDSR(Dataset):
    def __init__(self, data, blur_params, blur_params_an, max_noise, scale,
                 transforms=trans_none, prob_an=0.5):
        super(OemMultipleInputsOutputsDatasetBIDSR, self).__init__()

        if not isinstance(blur_params, list):
            raise NotListException

        if not isinstance(blur_params_an, list):
            raise NotListException

        if len(data) == 0 or len(blur_params) == 0 or len(blur_params_an) == 0:
            raise BadLenthException

        self.data = data
        self.max_noise, self.pn = max_noise
        self.blurs, self.blurs_pca, self.ps = blur_params
        self.blurs_an, self.blurs_pca_an, self.ps_an = blur_params_an
        self.transforms = transforms
        self.scale = scale
        self.idx_blur = np.arange(self.blurs.size(0))
        self.idx_blur_an = np.arange(self.blurs_an.size(0))
        #np.random.shuffle(self.idx_blur)
        #np.random.shuffle(self.idx_blur_an)
        #self.i_blur = 0
        #self.i_blur_an = 0
        self.prob_an = prob_an

    def __len__(self):
        return len(self.data)

    def get_kernel(self):
        gauss_b_idx = np.random.choice(self.idx_blur, 1, False, self.ps)[0]
        gauss_b = self.blurs[gauss_b_idx]
        params = self.blurs_pca[gauss_b_idx]

        return params, gauss_b

    def get_kernel_an(self):
        gauss_b_idx = np.random.choice(self.idx_blur_an, 1, False, self.ps_an)[0]
        gauss_b = self.blurs_an[gauss_b_idx]
        params = self.blurs_pca_an[gauss_b_idx]

        return params, gauss_b

    def get_random_kernels(self):
        if np.random.rand() >= self.prob_an:
            params_b, gauss_b = self.get_kernel()
        else:
            params_b, gauss_b = self.get_kernel_an()

        params = torch.zeros(params_b.size(0)+1)
        params[:-1] = params_b

        return params, gauss_b

    def get_noise(self, data, params):
        noise_var = uniform(0.0, self.max_noise)
        params[-1] = noise_var
        shape = [data.size(0), int(data.size(1)//self.scale), int(data.size(2)//self.scale)]
        noise = torch.zeros(shape).normal_(mean=0.0, std=noise_var)

        return noise

    def __getitem__(self, idx):
        x = self.data[idx]
        x, y = self.transforms(x, 0)
        x = torch.from_numpy(np.ascontiguousarray(x))
        #blur_params, blur_kernels = self.get_random_kernels()
        #noise_var = self.get_noise(x, blur_params)
        #data = [data.float()/255.0, blur_kernels, noise_var, blur_params]
        #data = [x, blur_kernels, noise_var, blur_params]
        #data = [x, blur_kernels, self.max_noise, self.pn, blur_params]
        data = [x, ]

        return data


class DatasetBSISR(Dataset):
    def __init__(self, data_hr, data_lr, data_lr_blur, pca, ps, transforms=trans_none):
        super(DatasetBSISR, self).__init__()
        self.data_hr = data_hr
        self.data_lr = data_lr
        self.data_lr_blur = data_lr_blur
        self.pca = pca
        self.ps = ps
        self.transforms = transforms
        self.idx_blur = np.arange(self.pca.size(0))

    def __len__(self):
        return len(self.data_hr)

    def __getitem__(self, idx):
        #gauss_b_idx = np.random.choice(self.idx_blur, 1, False, self.ps)[0]
        #x, y, y_no_blur = self.data_hr[idx], self.data_lr_blur[idx, gauss_b_idx], self.data_lr[idx]
        y = self.data_lr_blur[idx, 0]
        #pca = self.pca[gauss_b_idx]
        #x, y, y_no_blur = self.transforms(x, y, y_no_blur)
        #x = torch.from_numpy(np.ascontiguousarray(x))
        #y = torch.from_numpy(np.ascontiguousarray(y))
        #y_no_blur = torch.from_numpy(np.ascontiguousarray(y_no_blur))
        y = torch.from_numpy(y)
        #data = [y, y_no_blur, x, pca]

        return y
