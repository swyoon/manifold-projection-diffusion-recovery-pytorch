
import torch
import torch.nn as nn


class Mahalanobis(nn.Module):
    def __init__(self, maha_statistic_path, mode='MD'):
        super().__init__()
        self.mode = mode 

        maha_statistic = torch.load(maha_statistic_path)
        self.all_means = maha_statistic["all_means"].double()
        self.invcov = maha_statistic["invcov"].double()
        self.whole_mean = maha_statistic["whole_mean"].double()
        self.whole_invcov = maha_statistic["whole_invcov"].double()

    def forward(self, x):
        if self.mode == 'MD':
            return self.forward_maha(x)
        elif self.mode == 'RMD':
            return self.forward_maha(x) - self.forward_background_maha(x)
        else:
            raise NotImplementedError

    def predict(self, x):
        return self(x)

    def forward_maha(self, z, debug=False):
        """mahalanobis distance"""
        z = z.unsqueeze(-1)  # .double()
        z = z - self.all_means
        z = z
        op1 = torch.einsum("ijk,jl->ilk", z, self.invcov)
        op2 = torch.einsum("ijk,ijk->ik", op1, z)

        if debug:
            return op2, op1, z
        else:
            return torch.min(op2, dim=1).values.float()        

    def forward_background_maha(self, z):
        z = z - self.whole_mean
        op1 = torch.mm(z, self.whole_invcov)
        op2 = torch.mm(op1, z.t())
        return op2.diag().float()
