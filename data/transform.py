import torch

class LogCompression(object):
    """
    Performs log dynamic compression
    """
    def __init__(self,gamma=1000,min_clip = 1e-05):
        self.gamma = gamma
        self.min_clip = min_clip

    def __call__(self, sample):
        sample.clamp_(min=self.min_clip)
        return torch.log(1 + self.gamma * sample)

class InverseLogCompression(object):

    def __init__(self,gamma=1000):
        self.gamma = gamma

    def __call__(self, sample):
        torch.exp_(sample)
        sample -= 1
        return sample.div_(self.gamma)