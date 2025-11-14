# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import math
import torch
import torch.nn as nn
from .model_trans import CTransformer
from .ansatz_tq_cut import QEmbedding

hidden = 8

class CML_Model(nn.Module):
    def __init__(self, qnum_out, num_e, num_a, num_layer=6):
        super().__init__()
        self.qnum_out = qnum_out
        self.rpos = nn.Embedding(10, 3)
        self.embed_ae = nn.Linear(4, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.norm_out = nn.LayerNorm(hidden)
        self.activate = nn.Tanh()
        self.clayer = nn.ModuleList(
            [CTransformer(hidden, hidden * 4, 0) for _ in range(num_layer)])
        self.linear_int = nn.Linear(hidden, 2*hidden)
        self.norm_int = nn.LayerNorm(2*hidden)
        self.conv_a = nn.Conv1d(2*hidden, 2*hidden, 2, 2, bias=False)
        self.conv_e = nn.Conv1d(2*hidden, 2*hidden, 2, 2, bias=False)
        self.linear_out = nn.Linear(2*hidden, 2**qnum_out)
        self.qembed = QEmbedding(qnum_out)

    def forward(self, pos_a, ix_a, ix_e=None, bk_map=True):
        pos_ix = []
        atom_ix = []
        if ix_e is None:
            ix_e = ix_a
        for ix, ne in enumerate(ix_e):
            pos_ix += list(range(ne))
            atom_ix += [ix] * ne
        pos_a = torch.tensor(pos_a, dtype=torch.float).to(self.rpos.weight.device)
        pos_e = self.rpos(torch.tensor(pos_ix, dtype=torch.int).to(self.rpos.weight.device)) + pos_a[atom_ix]
        ae = pos_e.unsqueeze(1) - pos_a.unsqueeze(0)
        r_ae = torch.linalg.norm(ae, axis=2, keepdims=True, dtype=torch.float)
        seq = self.embed_ae(torch.cat((ae, r_ae), dim=-1))
        amp_proto = torch.tensor(ix_a).unsqueeze(-1).expand(seq.size()).to(seq.device)
        amp_ae = r_ae.std()
        bias_ae = r_ae.mean()

        for trans in self.clayer:
            seq = amp_proto * seq
            seq = trans(seq)
        ae_inv = torch.linalg.inv(self.embed_ae.weight.t() @ self.embed_ae.weight) @ self.embed_ae.weight.t()
        r_ae = (ae_inv.detach().unsqueeze(0).unsqueeze(0) @ seq.unsqueeze(-1))[:,:,-1]
        if torch.isnan(r_ae).any():
            print("NaN occurred")
            raise SystemExit
        r_ae = amp_ae * (r_ae - r_ae.mean()) / r_ae.std() + bias_ae
        x = self.linear_int(torch.exp(-r_ae) * amp_proto * seq).transpose(-2, -1)
        pad = torch.zeros((x.size(0), x.size(1), 1), device=x.device)
        y = torch.mean(x, dim=-1)
        amp_r = torch.mean(torch.exp(-r_ae.transpose(-2, -1)), dim=-1)
        for _ in range((x.size(-1)+1)//2):
            x = self.conv_a(torch.cat((x, pad), dim=-1))
        x = (amp_r * self.norm_int(y + x.squeeze(-1))).t()
        pad = torch.zeros((x.size(0), 1), device=x.device)
        y = torch.mean(x, dim=-1)
        amp_r = torch.mean(amp_r.t(), dim=-1)
        for _ in range((x.size(-1)+1)//2):
            x = self.conv_e(torch.cat((x, pad), dim=-1))
        x = amp_r * self.norm_int(y + x.squeeze(-1))
        psi = self.linear_out(x)
        if bk_map:
            hf = [math.pi, 0] * (len(pos_ix)//2) + [0] * (self.qnum_out - len(pos_ix))
        else:
            hf = [math.pi] * len(pos_ix) + [0] * (self.qnum_out - len(pos_ix))
        bos = self.qembed(torch.tensor(hf).unsqueeze(0).to(seq.device)).squeeze(0)

        return psi + bos * 2**(self.qnum_out/2)
