import numpy as np
import torch
import torch.nn as nn


class FadeToBlackMatrix(nn.Module):
    def __init__(self, im_size, kernel_size):
        super().__init__()
        self.pad_size = [kernel_size[0], kernel_size[1]]
        self.padding = nn.ReplicationPad2d(self.pad_size*2)
        im_size = np.array(im_size)
        im_size[-2:] = im_size[-2:] + np.array(self.pad_size)*2
        im_size = list(im_size)
        P_mat = torch.ones(im_size).unsqueeze(dim=0)
        P_mat = self.fadetoblack(P_mat)
        self.register_buffer('P_mat', P_mat)

    def pad(self, x):
        x = self.padding(x)*self.P_mat

        return x

    def unpad(self, x):
        x = x[..., self.pad_size[0]:-self.pad_size[0], self.pad_size[1]:-self.pad_size[1]].contiguous()

        return x

    def forward(self, x):
        return self.pad(x)

    def fadetoblack(self, P_mat):
        P_mat = P_mat.numpy().copy()

        #Up and bottom
        r_top = P_mat[..., self.pad_size[0], :].reshape(1, -1)
        r_bottom = P_mat[..., -self.pad_size[0]+1, :].reshape(1, -1)
        cs = np.linspace(0.0, 1.0, num=self.pad_size[0], endpoint=False).reshape(-1, 1)
        P_mat[..., :self.pad_size[0], :] = r_top*cs
        P_mat[..., -self.pad_size[0]:, :] = r_bottom*cs[..., ::-1, :]

        #Left and right
        c_left = P_mat[..., self.pad_size[1]].reshape(-1, 1)
        c_right = P_mat[..., -self.pad_size[1]+1].reshape(-1, 1)
        cs = np.linspace(0.0, 1.0, num=self.pad_size[1], endpoint=False).reshape(1, -1)
        P_mat[..., :self.pad_size[1]] = c_left*cs
        P_mat[..., -self.pad_size[1]:] = c_right*cs[..., ::-1]

        return torch.from_numpy(P_mat)


class FadeToBlackMatrixNoPad(nn.Module):
    def __init__(self, im_size, kernel_size):
        super().__init__()
        self.pad_size = [kernel_size[0], kernel_size[1]]
        im_size = np.array(im_size)
        im_size[-2:] = im_size[-2:]
        im_size = list(im_size)
        P_mat = torch.ones(im_size).unsqueeze(dim=0)
        P_mat = self.fadetoblack(P_mat)
        self.register_buffer('P_mat', P_mat)

    def pad(self, x):
        x = x*self.P_mat

        return x

    def unpad(self, x):
        x = x[..., self.pad_size[0]:-self.pad_size[0], self.pad_size[1]:-self.pad_size[1]].contiguous()

        return x

    def forward(self, x):
        return self.pad(x)

    def fadetoblack(self, P_mat):
        P_mat = P_mat.numpy().copy()

        #Up and bottom
        r_top = P_mat[..., self.pad_size[0], :].reshape(1, -1)
        r_bottom = P_mat[..., -self.pad_size[0]+1, :].reshape(1, -1)
        cs = np.linspace(0.0, 1.0, num=self.pad_size[0], endpoint=False).reshape(-1, 1)
        P_mat[..., :self.pad_size[0], :] = r_top*cs
        P_mat[..., -self.pad_size[0]:, :] = r_bottom*cs[..., ::-1, :]

        #Left and right
        c_left = P_mat[..., self.pad_size[1]].reshape(-1, 1)
        c_right = P_mat[..., -self.pad_size[1]+1].reshape(-1, 1)
        cs = np.linspace(0.0, 1.0, num=self.pad_size[1], endpoint=False).reshape(1, -1)
        P_mat[..., :self.pad_size[1]] = c_left*cs
        P_mat[..., -self.pad_size[1]:] = c_right*cs[..., ::-1]

        return torch.from_numpy(P_mat)
