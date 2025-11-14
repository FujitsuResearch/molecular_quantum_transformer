# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTransformer(nn.Module):
    def __init__(self, hidden, ff_hidden, num_layer=6):
        super().__init__()
        self.sa_block = SelfAttention(hidden)
        self.ff_block = nn.Sequential(
                            nn.Linear(hidden, ff_hidden),
                            nn.GELU(),
                            nn.Linear(ff_hidden, hidden)
                            )
        self.norm_sa = nn.LayerNorm(hidden)
        self.norm_ff = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.norm_sa(x + self.sa_block(x, x, x))
        x = self.norm_ff(x + self.ff_block(x))

        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.scale = math.sqrt(hidden)
        self.linear_q = nn.Linear(hidden, hidden)
        self.linear_k = nn.Linear(hidden, hidden)
        self.linear_v = nn.Linear(hidden, hidden)
        self.linear_o = nn.Linear(hidden, hidden)

    def forward(self, query, key, value):
        attn = torch.bmm(self.linear_q(query), self.linear_k(key).transpose(-1, -2)) / self.scale
        out = self.linear_o(torch.bmm(F.softmax(attn, dim=-1), self.linear_v(value)))

        return out


