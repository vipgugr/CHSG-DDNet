import torch
from torch.autograd import Variable


class Charbonnier():
    def __init__(self, epsilon=0.001, size_average=True):
        self.epsilon = epsilon**2
        self.size_average = size_average
    
    def __call__(self, x, y):
        l = torch.sqrt((x-y)**2 + self.epsilon)
        l = l.mean() if self.size_average else l.sum()
        
        return l
