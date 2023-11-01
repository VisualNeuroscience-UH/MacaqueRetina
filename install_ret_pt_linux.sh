#!/bin/bash

# Install packages with pip
pip install ipython
pip install torch torchvision torchaudio
pip install brian2
pip install notebook jupyterlab scikit-learn pytest scikit-image matplotlib tqdm seaborn
pip install opencv-python-headless
pip install colorednoise
pip install h5py
pip install black
pip install "ray[tune]"
pip install torch-fidelity torchmetrics
pip install optuna
pip install torchsummary
pip install pyshortcuts
pip install Shapely

# Go one directory up from MacaqueRetina repo
cd .. 

# Clone cxsystem2 repo and install
git clone https://github.com/VisualNeuroscience-UH/CxSystem2.git
cd CxSystem2
pip install -U -e .

# Confirm installation
cxsystem2

cd ../MacaqueRetina

# Deactivate the virtual environment
deactivate

