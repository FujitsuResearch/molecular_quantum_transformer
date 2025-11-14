# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import os, sys, pickle, random
import argparse
from tqdm import tqdm

import pennylane as qml
from pennylane import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

num_proc = 4
list_comp = ["H2", "LiH", "BeH2", "H4"]

### Select target molecule by index num. in list_comp ###
ix_comp = 1

comp = list_comp[ix_comp]
if ix_comp == 0:
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.400287]])
    kwargs_H = {'basis':'6-31g', 'method':'pyscf', 'mapping':'bravyi_kitaev'}
elif ix_comp == 1:
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.969280527]])
    kwargs_H = {'basis':'sto-3g', 'method':'pyscf', 'mapping':'bravyi_kitaev'}
elif ix_comp == 2:
    symbols = ["H", "Be", "H"]
    coordinates = np.array([[0.0, 0.0, -2.969280527], [0.0, 0.0, 0.0], [0.0, 0.0, 2.969280527]])
    kwargs_H = {'basis':'sto-3g', 'method':'pyscf', 'mapping':'bravyi_kitaev'}
elif ix_comp == 3:
    symbols = ["H", "H", "H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.324363], [2.324363, 0.0, 0.0], [2.324363, 0.0, 2.324363]])
    kwargs_H = {'basis':'sto-3g'}
is_bk = (kwargs_H.get('mapping') == 'bravyi_kitaev')

mol = qml.qchem.Molecule(symbols, coordinates)
data_H = qml.qchem.molecular_hamiltonian(symbols, coordinates, **kwargs_H)
qnum_out = data_H[1]
print("qubits: {}".format(qnum_out))

dev = qml.device("default.qubit", wires=qnum_out)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def cost_fn(psi, symbols, coordinates, lock):
    qml.AmplitudeEmbedding(psi, wires=range(qnum_out), pad_with=0., normalize=True)
    lock.acquire()
    H = qml.qchem.molecular_hamiltonian(symbols, coordinates, **kwargs_H)[0]
    lock.release()
    return qml.expval(H)


def train(model, rank, lock):
    optim = AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999), weight_decay=0.001)
    sched = StepLR(optim, step_size=100, gamma=0.9)

    with tqdm(range(int(10000 / num_proc))) as pbar:
        for epoch in pbar:
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

            psi = model(coordinates, mol.nuclear_charges, bk_map=is_bk)
            loss = cost_fn(psi, symbols, coordinates, lock)
            loss.backward()
            optim.step()
#            sched.step()
            model.zero_grad()
            pbar.set_postfix(loss=loss.item())


def test(model):
    curve_ene = []
    curve_ext = []
    for i in range(1, 50):
        if ix_comp == 0 or ix_comp == 1:
            #### H2, LiH  ####
            coordinates[1, 2] = 0.1 * i
        elif ix_comp == 2:
            #### BeH2 ####
            coordinates[2, 2] = 0.1 * i
            coordinates[0, 2] = -coordinates[2, 2]
        elif ix_comp == 3:
            #### H4 ###
            coordinates[2, 0] = 0.1 * i
            coordinates[3, 0] = coordinates[2, 0]

        psi = model(coordinates, mol.nuclear_charges, bk_map=is_bk)
        curve_ene.append(cost_fn(psi, symbols, coordinates, lock).item())
        H = qml.qchem.molecular_hamiltonian(symbols, coordinates, **kwargs_H)
        eig = qml.eigvals(qml.SparseHamiltonian(H[0].sparse_matrix(), range(qnum_out)))
        curve_ext.append(eig.item())
    print("\nloss = {}".format(curve_ene))
    print("\neigen = {}".format(curve_ext))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cml", action='store_true')

    args = parser.parse_args()

    if args.cml:
        print("Classical Transformers")
        from model.chem_model_cml import CML_Model
        model = CML_Model(qnum_out, mol.n_electrons, len(mol.symbols))
    else:
        print("Quantum Transformers")
        from model.chem_model_qattn import QML_Model
        model = QML_Model(qnum_out, mol.n_electrons, len(mol.symbols))
    model.to(torch.device("cuda"))
    model.train()
    model.share_memory()

    processes = []
    mp.set_start_method('spawn', force=True)
    lock = mp.Lock()
    for rank in range(num_proc):
        p = mp.Process(target=train, args=(model, rank, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("** ** * Saving model * ** **")
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )

    path_save = "trained"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    output_model_file = os.path.join(path_save, "{}_diff_bond_model.pt".format(comp))
    torch.save(model_to_save.state_dict(), output_model_file)

    test(model.eval())
