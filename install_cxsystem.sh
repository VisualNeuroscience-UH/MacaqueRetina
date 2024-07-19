#!/bin/bash

# Optional dependencies

# Clone cxsystem2 repo and install
cd .. 
git clone https://github.com/VisualNeuroscience-UH/CxSystem2.git
cd CxSystem2
pip install -U -e .

# Confirm installation
cxsystem2

cd ../MacaqueRetina