import torch


class CharbonnierLoss():
    def __init__(self, eps=1e-3):
        self.eps= eps

    def __call__(self, pred, target):
        pred, _ = pred
        loss = torch.sqrt((pred - target)**2 + self.eps).mean()

        return loss