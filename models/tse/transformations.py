import numpy as np
from torch import from_numpy
import random


def trans_none(x, y, z=None):
    if z is None:
        return x, y
    else:
        return x, y, z

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y, z=None):
        for t in self.transforms:
            if z is None:
                x, y = t(x, y)
            else:
                x, y, z = t(x, y, z)

        if z is None:
            return x, y
        else:
            return x, y, z

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomCrop(object):
    def __init__(self, size, trans_label=False):
        self.size = size
        self.trans_label = trans_label

    @staticmethod
    def get_params(x, output_size):
        w, h = x.shape[1], x.shape[2]
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, i+th, j+tw

    def __call__(self, x, y, z=None):
        i, j, h, w = self.get_params(x, self.size)
        x = x[:, i:h, j:w]#.contiguous()

        if self.trans_label:
            y = y[:, i:h, j:w]#.contiguous()

            if not (z is None):
                z = z[:, i:h, j:w]

        if z is None:
            return x, y
        else:
            return x, y, z


class RandomCropSR(object):
    def __init__(self, size, factor, trans_label=False):
        self.size = size
        self.factor = factor
        self.trans_label = trans_label

    def get_params(self, x, output_size):
        w, h = x.shape[1], x.shape[2]
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        hr_size = [i, j, i+th, j+tw]
        lr_size = [int(hr_size[0]//self.factor), int(hr_size[1]//self.factor), int(hr_size[2]//self.factor), int(hr_size[3]//self.factor)]

        return hr_size, lr_size

    def __call__(self, x, y, z=None):
        hr_size, lr_size = self.get_params(x, self.size)
        i, j, h, w = hr_size
        x = x[:, i:h, j:w]#.contiguous()

        if self.trans_label:
            i, j, h, w = lr_size
            y = y[:, i:h, j:w]#.contiguous()

            if not (z is None):
                z = z[:, i:h, j:w]

        if z is None:
            return x, y
        else:
            return x, y, z


class FlipVerHor(object):
    def __init__(self, trans_label=False):
        self.trans_label = trans_label
        self.ks = np.array([0, 1, 2], np.int32)

    def __call__(self, x, y, z=None):
        k = np.random.choice(self.ks)
        if k==0:
            x = x[:, ::-1, :]

            if self.trans_label:
                y = y[:, ::-1, :]

                if not (z is None):
                    z = z[:, ::-1, :]

        elif k==1:
            x = x[:, :, ::-1]

            if self.trans_label:
                y = y[:, :, ::-1]

                if not (z is None):
                    z = z[:, :, ::-1]

        if z is None:
            return x, y
        else:
            return x, y, z


class FlipVer(object):
    def __init__(self, trans_label=False):
        self.trans_label = trans_label

    def __call__(self, x, y, z=None):
        if random.random() < 0.5:
            x = x[:, ::-1, :]

            if self.trans_label:
                y = y[:, ::-1, :]

                if not (z is None):
                    z = z[:, ::-1, :]

        if z is None:
            return x, y
        else:
            return x, y, z


class FlipHor(object):
    def __init__(self, trans_label=False):
        self.trans_label = trans_label

    def __call__(self, x, y, z=None):
        if random.random() < 0.5:
            x = x[:, :, ::-1]

            if self.trans_label:
                y = y[:, :, ::-1]

                if not (z is None):
                    z = z[:, :, ::-1]

        if z is None:
            return x, y
        else:
            return x, y, z


class RandomRotation(object):
    def __init__(self, ks, trans_label):
        if len(ks) < 1:
            raise ValueError("Too few samples.")

        self.ks = ks
        self.trans_label = trans_label

    @staticmethod
    def get_params(ks):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        k = np.random.choice(ks)

        return k

    def __call__(self, x, y, z=None):
        k = self.get_params(self.ks)
        x = np.rot90(x, k=k, axes=(1, 2))

        if self.trans_label:
            y = np.rot90(y, k=k, axes=(1, 2))

            if not (z is None):
                z = np.rot90(z, k=k, axes=(1, 2))

        if z is None:
            return x, y
        else:
            return x, y, z

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)
