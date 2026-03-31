import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv
from ..modules.block import *
from ..modules.block import C3k, PSABlock





class C3k2_v2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2PSA_v2(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5) -> None:
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        a, b = self.cv0(x), self.cv1(x)
        b = self.m(b)
        return self.cv2(torch.cat([a, b], 1))

