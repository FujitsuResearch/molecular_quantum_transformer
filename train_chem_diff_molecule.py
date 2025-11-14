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
comp = "LiH"
list_symbols = \
    [
        ["H", "H"],
        ["H", "Be", "H"],
        ["H", "H", "H", "H"],
    ]
list_coordinates = \
    [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.400287]]),
        np.array([[0.0, 0.0, -2.969280527], [0.0, 0.0, 0.0], [0.0, 0.0, 2.969280527]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.324363], [2.324363, 0.0, 0.0], [2.324363, 0.0, 2.324363]]),
    ]
kwargs_H = {'basis':'sto-3g', 'method':'pyscf', 'mapping':'bravyi_kitaev'}

target_symbols = ["Li", "H"]
target_coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.969280527]])
target_kwargs_H = {'basis':'sto-3g', 'method':'pyscf', 'mapping':'bravyi_kitaev'}

mol = qml.qchem.Molecule(list_symbols[1], list_coordinates[1])
data_H = qml.qchem.molecular_hamiltonian(list_symbols[1], list_coordinates[1], **kwargs_H)
qnum_out = data_H[1]
print("qubits: {}".format(qnum_out))

dev = qml.device("default.qubit", wires=qnum_out)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def cost_fn(psi, symbols, coordinates, charge, kwargs_H, lock):
    qml.AmplitudeEmbedding(psi, wires=range(qnum_out), pad_with=0., normalize=True)
    lock.acquire()
    H = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=charge, **kwargs_H)[0]
    lock.release()
    return qml.expval(H)


def train(model, rank, lock):
    optim = AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999), weight_decay=0.001)

    with tqdm(range(int(20000 / num_proc))) as pbar:
        for epoch in pbar:
            ix = random.randint(0, 2)
            symbols = list_symbols[ix]
            coordinates = list_coordinates[ix]
            charge = 0
            if ix == 0:
                #### H2, LiH  ####
                coordinates[1, 2] = random.uniform(1e-5, 5.0)
            elif ix == 1:
                #### BeH2 ####
                coordinates[2, 2] = random.uniform(1e-5, 5.0)
                coordinates[0, 2] = -coordinates[2, 2]
            elif ix == 2:
                #### H4 ###
                coordinates[2, 0] = random.uniform(1e-5, 5.0)
                coordinates[3, 0] = coordinates[2, 0]

            mol = qml.qchem.Molecule(symbols, coordinates, charge)
            ix_e = mol.nuclear_charges
            psi = model(coordinates, mol.nuclear_charges, ix_e)
            loss = cost_fn(psi, symbols, coordinates, charge, kwargs_H, lock)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            weight = model.rpos.weight.data.clone()
            optim.step()
            model.zero_grad()
            pbar.set_postfix(loss=loss.item())


def test(model):
    curve_ene = []
    curve_ext = []
    for i in range(1, 50):
        target_coordinates[1, 2] = 0.1 * i
        mol = qml.qchem.Molecule(target_symbols, target_coordinates)
        ix_e = mol.nuclear_charges
        psi = model(target_coordinates, mol.nuclear_charges, ix_e)
        curve_ene.append(cost_fn(psi, target_symbols, target_coordinates, 0, target_kwargs_H, lock).item())
        H = qml.qchem.molecular_hamiltonian(target_symbols, target_coordinates, **target_kwargs_H)
        eig = qml.eigvals(qml.SparseHamiltonian(H[0].sparse_matrix(), range(H[1])))
        curve_ext.append(eig.item())
    print("\nloss = {}".format(curve_ene))
    print("\neigen = {}".format(curve_ext))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cml", action='store_true')
    parser.add_argument("--eval", action='store_true')

    args = parser.parse_args()

    if args.cml:
        print("Classical Transformers")
        from model.chem_model_cml import CML_Model
        model = CML_Model(qnum_out, mol.n_electrons, len(mol.symbols), 12)
    else:
        print("Quantum Transformers")
        from model.chem_model_qattn import QML_Model
        model = QML_Model(qnum_out, mol.n_electrons, len(mol.symbols), 12)

    if args.eval:
        model.load_state_dict(torch.load('trained/LiH_diff_molecle_model.pt'))
    model.to(torch.device("cuda"))
    model.train()
    model.share_memory()

    processes = []
    mp.set_start_method('spawn', force=True)
    lock = mp.Lock()
    if not args.eval:
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
        output_model_file = os.path.join(path_save, "{}_diff_molecle_model.pt".format(comp))
        torch.save(model_to_save.state_dict(), output_model_file)

    test(model.eval())
