import xarray as xr
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalization, pointwise gaussian
class DataNormalizer(object):
    def __init__(self, s):
        super(DataNormalizer, self).__init__()
        self.s = torch.tensor(s)

    def encode(self, x):
        x = x / self.s
        return x

    def decode(self, x):
        x = x * self.s
        return x

    def to(self, device):
        if torch.is_tensor(self.s):
            self.s = self.mean.to(device)
        else:
            self.s = torch.from_numpy(self.s).to(device)
        return self

    def cuda(self):
        self.s = self.s.cuda()

    def cpu(self):
        self.s = self.s.cpu()

#x_normalizer = DataNormalizer(np.array([1e3, 10, 10], dtype=np.float32))
#y_normalizer = DataNormalizer(np.array([1e3, 1e3], dtype = np.float32))



#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
