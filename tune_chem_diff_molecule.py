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

qnum_out = 14
num_proc = 4
comp = "LiH"

symbols = ["Li", "H"]
list_coordinates = \
    [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.5]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.5]]),
    ]
kwargs_H = {'basis':'sto-3g', 'method':'pyscf', 'mapping':'bravyi_kitaev'}

mol = qml.qchem.Molecule(symbols, list_coordinates[0])
print("qubits: {}".format(qnum_out))


dev = qml.device("default.qubit", wires=qnum_out)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def cost_fn(psi, symbols, coordinates, kwargs_H, lock, H=None):
    qml.AmplitudeEmbedding(psi, wires=range(qnum_out), pad_with=0., normalize=True)
    if H is None:
        lock.acquire()
        H = qml.qchem.molecular_hamiltonian(symbols, coordinates, **kwargs_H)[0]
        lock.release()
    return qml.expval(H)


def train(model, rank, lock):
    optim = AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999), weight_decay=0.001)

    list_H = []
    for target_coordinates in list_coordinates:
        H = qml.qchem.molecular_hamiltonian(symbols, target_coordinates, **kwargs_H)[0]
        list_H.append(H)

    with tqdm(range(int(2000 / num_proc))) as pbar:
        for epoch in pbar:
            ix = random.randint(0, 4)
            psi = model(list_coordinates[ix], mol.nuclear_charges, mol.nuclear_charges)
            loss = cost_fn(psi, symbols, list_coordinates[ix], kwargs_H, lock, list_H[ix])
            loss.backward()
            optim.step()
            model.zero_grad()
            pbar.set_postfix(loss=loss.item())


def test(model):
    curve_ene = []
    curve_ext = []
    for i in range(1, 50):
        target_coordinates = list_coordinates[0]
        target_coordinates[1, 2] = 0.1 * i
        mol = qml.qchem.Molecule(symbols, target_coordinates)
        psi = model(target_coordinates, mol.nuclear_charges, mol.nuclear_charges)
        curve_ene.append(cost_fn(psi, symbols, target_coordinates, kwargs_H, lock).item())
        H = qml.qchem.molecular_hamiltonian(symbols, target_coordinates, **kwargs_H)
        eig = qml.eigvals(qml.SparseHamiltonian(H[0].sparse_matrix(), range(H[1])))
        curve_ext.append(eig.item())
    print("\nloss = {}".format(curve_ene))
    print("\neigen = {}".format(curve_ext))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cml", action='store_true')
    parser.add_argument("--scratch", action='store_true')
    parser.add_argument("--eval", action='store_true')

    args = parser.parse_args()

    if args.cml:
        print("Classical Transformers")
        from model.chem_model_proto_cml_v3 import CML_Model
        model = CML_Model(qnum_out, mol.n_electrons, len(mol.symbols), 12)
    else:
        print("Quantum Transformers")
        from model.chem_model_proto_qattn_v3 import QML_Model
        model = QML_Model(qnum_out, mol.n_electrons, len(mol.symbols), 12)

    if args.scratch:
        print("Scratch")
        pass
    elif args.eval:
        model.load_state_dict(torch.load('tuned/LiH_diff_molecle_model.pt'))
    else:
        print("Finetuning")
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

        path_save = "tuned"
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        output_model_file = os.path.join(path_save, "{}_diff_molecle_model.pt".format(comp))
        torch.save(model_to_save.state_dict(), output_model_file)

    test(model.eval())
