# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import os, sys, pickle, random
import argparse
from tqdm import tqdm

import pennylane as qml
from pennylane import numpy as np


list_comp = ["H2", "LiH", "BeH2", "H4"]

### Select target molecule by index num. in list_comp ###
ix_comp = 1

comp = list_comp[ix_comp]
if ix_comp == 0:
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.400287]])
    kwargs_H = {'basis':'6-31g'}
elif ix_comp == 1:
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.969280527]])
    kwargs_H = {'basis':'sto-3g'}
elif ix_comp == 2:
    symbols = ["H", "Be", "H"]
    coordinates = np.array([[0.0, 0.0, -2.969280527], [0.0, 0.0, 0.0], [0.0, 0.0, 2.969280527]])
    kwargs_H = {'basis':'sto-3g'}
elif ix_comp == 3:
    symbols = ["H", "H", "H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.324363], [2.324363, 0.0, 0.0], [2.324363, 0.0, 2.324363]])
    kwargs_H = {'basis':'sto-3g'}

mol = qml.qchem.Molecule(symbols, coordinates)
data_H = qml.qchem.molecular_hamiltonian(symbols, coordinates, **kwargs_H)
qnum_out = data_H[1]
print("qubits: {}".format(qnum_out))

dev = qml.device("default.qubit", wires=qnum_out)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def cost_fn(state, symbols, coordinates):
    qml.BasisState(state, wires=range(qnum_out))
    H = qml.qchem.molecular_hamiltonian(symbols, coordinates, **kwargs_H)[0]
    return qml.expval(H)


if __name__ == '__main__':
    curve_hf = []
    for i in range(1, 50):
        if ix_comp == 0 or ix_comp == 1:
            #### H2, LiH  ####
            coordinates[1, 2] = random.uniform(1e-5, 5.0)
        elif ix_comp == 2:
            #### BeH2 ####
            coordinates[2, 2] = random.uniform(1e-5, 5.0)
            coordinates[0, 2] = -coordinates[2, 2]
        elif ix_comp == 3:
            #### H4 ###
            coordinates[2, 0] = random.uniform(1e-5, 5.0)
            coordinates[3, 0] = coordinates[2, 0]

        state = qml.qchem.hf_state(mol.n_electrons, qnum_out)
        curve_hf.append(cost_fn(state, symbols, coordinates).item())
    print("hf = {}".format(curve_hf))
