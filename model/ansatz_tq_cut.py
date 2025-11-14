# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq


class QEmbedding(nn.Module):
    def __init__(self, qnum):
        super().__init__()
        self.qnum = qnum
        self.q_embed = tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i]}
            for i in range(qnum)])

    def forward(self, tokens):
        q_device = tq.QuantumDevice(n_wires=self.qnum, bsz=tokens.size(0), device=tokens.device)
        self.q_embed(q_device, tokens)

        return q_device.get_states_1d()


class QMeasurement(nn.Module):
    def __init__(self, qnum):
        super().__init__()
        self.qnum = qnum
        self.q_embed = tq.StateEncoder()
        self.meas = tq.MeasureAll(tq.PauliZ)

    def forward(self, phi):
        q_device = tq.QuantumDevice(n_wires=self.qnum, bsz=phi.size(0), device=phi.device)
        self.q_embed(q_device, phi)

        return self.meas(q_device)


class QSelfAttention(nn.Module):
    def __init__(self, qnum):
        super().__init__()
        self.qnum = qnum
        self.v_embed = tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i]}
            for i in range(qnum)])
        self.q_embed = tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i+1]}
            for i in range(qnum)])
        self.k_embed = tq.GeneralEncoder([{'input_idx': [i], 'func': 'ry', 'wires': [i+1]}
            for i in range(qnum)])
        self.out_embed = tq.StateEncoder()
        self.v_param = nn.Parameter(torch.Tensor(6, 3, qnum))
        self.qk_param = nn.Parameter(torch.Tensor(3, qnum))
        self.attn_param = nn.Parameter(torch.Tensor(2, qnum))
        nn.init.normal_(self.v_param, 0, math.pi/3)
        nn.init.normal_(self.qk_param, 0, math.pi/3)
        nn.init.normal_(self.attn_param, 0, math.pi/3)

        self.h = tq.Hadamard()
        self.x = tq.PauliX()
        self.rz = tq.RZ()
        self.ry = tq.RY()
        self.cnot = tq.CNOT()
        self.cz = tq.CZ()
        self.cswap = tq.CSWAP()
        self.cry = tq.CRY()
        self.cu3 = tq.CU3()
        self.reset = tq.Reset()
        self.meas = tq.MeasureAll(tq.PauliZ)

    def forward(self, query, key, value):
        def inner_strent(dev, wires, params, repeat=1):
            for n in range(repeat):
                for ix, w in enumerate(wires):
                    self.rz(dev, wires=w, params=params[n][0][ix])
                    self.ry(dev, wires=w, params=params[n][1][ix])
                    self.rz(dev, wires=w, params=params[n][2][ix])
                for ix in range(len(wires)):
                    tg = (ix+1 + n) % len(wires)
                    if ix == tg:
                        tg = (tg+1) % len(wires)
                    self.cnot(dev, wires=[wires[ix], wires[tg]])

        def inner_mps(dev, wires, params, repeat=1):
            for n in range(repeat):
                for ix in range(len(wires)-1):
                    self.rz(dev, wires=wires[ix], params=params[n][0][2*ix])
                    self.ry(dev, wires=wires[ix], params=params[n][1][2*ix])
                    self.rz(dev, wires=wires[ix], params=params[n][2][2*ix])
                    self.rz(dev, wires=wires[ix+1], params=params[n][0][2*ix+1])
                    self.ry(dev, wires=wires[ix+1], params=params[n][1][2*ix+1])
                    self.rz(dev, wires=wires[ix+1], params=params[n][2][2*ix+1])
                    self.cnot(dev, wires=[wires[ix], wires[ix+1]])

        num_batch = query.size(0)
        num_query = query.size(1)
        num_key = key.size(1)
        value = value.unsqueeze(1).repeat(1, num_query, 1, 1).view(-1, self.qnum)
        query = query.unsqueeze(2).repeat(1, 1, num_key, 1).view(-1, self.qnum)
        key = key.unsqueeze(1).repeat(1, num_query, 1, 1).view(-1, self.qnum)

        v_device = tq.QuantumDevice(n_wires=self.qnum, bsz=value.size(0), device=value.device)
        #### MPS attention #####
        value_new = v_device.get_states_1d().view(num_batch, num_query, num_key, -1)[:, :, 0]
        self.v_embed(v_device, value)
        inner_strent(v_device, range(self.qnum), self.v_param, 6)

        attn_device = tq.QuantumDevice(n_wires=self.qnum+1, bsz=query.size(0), device=query.device)
        #### MPS attention #####
        self.q_embed(attn_device, query)
        for ix in range(self.qnum):
            self.rz(attn_device, wires=ix+1, params=self.qk_param[0][ix])
            self.ry(attn_device, wires=ix+1, params=self.qk_param[1][ix])
        self.k_embed(attn_device, -key)
        self.x(attn_device, wires=0)
        for ix in range(self.qnum-1):
            self.ry(attn_device, wires=ix+1, params=self.attn_param[0][ix])
            self.ry(attn_device, wires=ix+2, params=self.attn_param[1][ix])
            self.cnot(attn_device, wires=[ix+1, ix+2])
        self.ry(attn_device, wires=self.qnum, params=self.attn_param[0][self.qnum-1])
        self.ry(attn_device, wires=0, params=self.attn_param[1][self.qnum-1])
        self.cnot(attn_device, wires=[self.qnum, 0])

        for ix in range(1, self.qnum+1):
            self.reset(attn_device, ix)

        attn_val = attn_device.get_states_1d().view(num_batch, num_query, num_key, 2, -1)
        v_val = v_device.get_states_1d().view(num_batch, num_query, num_key, -1)
        for ix in range(num_key):
            value_new = attn_val[:, :, ix, 0, 0].unsqueeze(-1) * value_new \
                + attn_val[:, :, ix, 1, 0].unsqueeze(-1) * v_val[:, :, ix]

        value_new = value_new.view(-1, 2**self.qnum)
        out_device = tq.QuantumDevice(n_wires=self.qnum, bsz=value_new.size(0), device=value_new.device)
        self.out_embed(out_device, value_new)

        return self.meas(out_device).view(-1, num_query, self.qnum)
