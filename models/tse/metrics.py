import torch
from torch.autograd import Variable
from torch.nn import L1Loss, MSELoss
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def Accuracy(x, y):
    batch_size = x.size(0)
    predictions = x.max(1)[1].type_as(y)
    predictions = predictions.view(-1)
    correct = predictions.eq(y.view(-1))
    if not hasattr(correct, 'sum'):
        correct = correct.cpu()
    correct = correct.type(torch.FloatTensor).sum()

    return (100. * correct) / batch_size


class Accuracy_multiclass():
    def __init__(self, nclass):
        self.nclass = nclass

    def __call__(self, x, y):
        batch_size = x.size(0)
        predictions = x.max(1)[1].type_as(y)
        correct = predictions.eq(y)
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.type(torch.FloatTensor)
        out = torch.zeros(2)

        y_i = (y==self.nclass).cpu()
        n_i = y_i.sum().item()
        out[0] += correct[y_i].sum().item()
        out[1] += float(n_i)

        return Variable(out)

    def combine(self, scores):
        return 100.0 * scores[0]/scores[1] if scores[1] != 0 else 100


class AUC():
    def __init__(self):
        self.py = []
        self.y = []

    def __call__(self, py, y):
        if y.is_cuda:
            self.y.append(y.data.cpu().numpy())
            self.py.append(py[:, 1].data.cpu().numpy())
        else:
            self.y.append(y.data.numpy())
            self.py.append(py[:, 1].data.numpy())

        return Variable(torch.zeros(2))

    def combine(self, scores):
        self.py = np.concatenate(self.py, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        auc = 100.0 * roc_auc_score(self.y, self.py)
        self.py = []
        self.y = []

        return auc


class F1():
    def __init__(self):
        self.py = []
        self.y = []
        self.ths = np.arange(0.0, 1.01, 0.01)

    def __call__(self, py, y):
        if y.is_cuda:
            self.y.append(y.data.cpu().numpy())
            self.py.append(py[:, 1].data.cpu().numpy())
        else:
            self.y.append(y.data.numpy())
            self.py.append(py[:, 1].data.numpy())

        return Variable(torch.zeros(2))

    def combine(self, scores):
        self.py = np.concatenate(self.py, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        f1_scores_ths = np.array([f1_score(self.y, self.py>th) for th in self.ths], np.float64)
        f1 = 100.0 * f1_scores_ths.max()
        self.py = []
        self.y = []

        return f1


class Accuracy_hinge_multiclass():
    def __init__(self, nclass):
        self.nclass = nclass

    def __call__(self, x, y):
        batch_size = x.size(0)
        predictions = torch.sign(x).type_as(y)[:, 0]
        correct = predictions.eq(y)

        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.type(torch.FloatTensor)
        out = torch.zeros(2)

        y_i = (y==self.nclass).cpu()
        n_i = y_i.sum().item()
        out[0] += correct[y_i].sum().item()
        out[1] += n_i

        return Variable(out)

    def combine(self, scores):
        return 100.0 * scores[0]/scores[1] if scores[1] != 0 else 100


def PSNR(x, y, max_val=1.0):
    mse = ((x-y)**2).mean()
    return -10*torch.log10(max_val/mse)


class MSE_Sub_Vec():
    def __init__(self, ini=0, end=0, size_average=True):
        self.ini = ini if ini >= 0 else 0
        self.end = end if end > ini else 0
        self.size_average = size_average

    def __call__(self, x, y):
        end = self.end if end > 0 else x.size(1)
        out = (x[ini:end]-y[ini:end])**2
        out = out.mean() if self.size_average else out.sum()

        return out


class L1_Sub_Vec():
    def __init__(self, ini=0, end=0, size_average=True):
        self.ini = ini if ini >= 0 else 0
        self.end = end if end > ini else 0
        self.l1 = L1Loss(size_average=size_average)

    def __call__(self, x, y):
        ini = self.ini
        end = self.end if self.end > 0 else x.size(1)
        out = self.l1(x[ini:end], y[ini:end])

        return out
