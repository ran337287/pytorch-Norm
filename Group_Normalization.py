##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: ran337287
## Email: belongtoabcd@126.com
## Copyright (c) 2018
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

class GroupNormND(_BatchNorm):
    def __init__(self, in_channels, groups, eps=1e-5, affine=True, momentum=0.1):
        """
        :param in_channels: [int] channel of input feature maps. (B, C, H, W) is C.
        :param groups: [int or None] the group number. If G is set to int, then channels_per_group must be None.
        :param eps: eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        :param affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
        :param momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        """
        super(GroupNormND, self).__init__(groups, eps, momentum, affine)

        assert not groups is None

        if groups is not None:
            assert in_channels % groups == 0
        self.groups = groups

    def forward(self, x):
        # print x
        input_size = x.size()
        N, C= input_size[:2]

        if self.groups is not None:
            groups = self.groups

        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)

        weight, bias = None, None
        if self.affine:
            weight = self.weight.repeat(N)
            bias = self.bias.repeat(N)

        x_reshaped = x.contiguous().view(1, N*groups, -1)
        out = F.batch_norm(x_reshaped, running_mean, running_var, weight, bias,
                           not self.use_running_stats, self.momentum, self.eps)

        self.running_mean.copy_(running_mean.view(N, groups).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(N,groups).mean(0, keepdim=False))

        return out.view(input_size)

    def use_running_stats(self, mode=True):
        self.use_running_stats = mode


class GroupNorm1D(GroupNormND):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))
        super(GroupNorm1D, self)._check_input_dim(input)

class GroupNorm2D(GroupNormND):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(GroupNorm2D, self)._check_input_dim(input)

class GroupNorm3D(GroupNormND):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(GroupNorm3D, self)._check_input_dim(input)

if __name__ == '__main__':

    x = torch.autograd.Variable(torch.randn(2, 4, 4, 4))
    BN = GroupNorm2D(in_channels=4, groups=2)
    out = BN(x)
    print out


