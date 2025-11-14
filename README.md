# Molecular Quantum Transformer
This repository is the official PyTorch implementation of the paper [Molecular Quantum Transformer](https://arxiv.org/abs/2503.21686).

## Installation
Download this reposistry [MQT](https://github.com/FujitsuResearch/molecular_quantum_transformer).

Install and setup CUDA Toolkit 12.4 Update 1 and Python 3.12.4.
Install required packages with utilizing 'requirements.txt'.

## Estimation of the potential energy curves with plain training
To select a target molecule, set the index number of the molecule list in the source code 'train_chem_diff_bond.py'. 
```shell
python3 train_chem_diff_bond.py 
```
When you train the model with classical Transformers, add the argument '--cml' to this command.

## Estimation of the potential energy curves with pretraining
1. Pretraining
```shell
python3 train_chem_diff_molecule.py
```
2. Few-shot learning
```shell
python3 tune_chem_diff_molecule.py 
```
When you train the model with classical Transformers, add the argument '--cml' to these commands.
If you want to fine-tune the model without pre-training (i.e. learn from scratch), run the command with the argument '--scratch'.

## License

This project is licensed under the terms of the BSD 3-Clause Clear License. Copyright 2025 Fujitsu Limited. All Rights Reserved.
