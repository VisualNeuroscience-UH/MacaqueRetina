# MacaqueRetina

Python software to build model of macaque retina and convert various visual stimuli to ganglion cell action potentials

This project is under development.

## Setup

- You need python 3.11 or above in your base environment  
- You need git installed.  
- Add the MacaqueRetina git repository root (the folder where this file resides) to the PYTHONPATH environment variable.

If you have NVIDIA GPU, you can install CUDA for using GPU acceleration. This is available both on [WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and  [linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)  
You can work with just CPU, but training new VAE models is slow. Luckily the standard VAE model is rather small, taking only about an hour on CPU in a fast laptop.

### Install with Windows

The python environment had issues in pure Windows, thus it is only supported under linux.  

- Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows subsystem for Linux) to your Windows system.

Put .wslconfig named file to your home directory in Windows to adjust memory etc. Example lines:
```
[wsl2]
memory=20GB
processors=8
swap=8GB
```

**Make sure to apply WSL2 config changes by restarting WSL2 or your machine**.  
More examples and backgorund on WSL2 in [Microsoft pages](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

Next, follow the installation in linux, when you have the WSL2 up and you are in linux system.

## Install with Linux

Run the following commands to create and activate virtual environment. You need python3 in your system path.

`python3 -m venv [your path]/ret_pt`
  - Make a new virtual environment named *ret_pt*, for retina and pytorch. Feel free to select own environment name.

`source [your root]/ret_pt/bin/activate`
  - Activate the virtual environment

### Install pytorch

Pytorch is integrated in MacaqueRetina, because it enables GPU acceleration when CUDA device is available.  
Get your system- and cuda/cpu-specific [installation command for Pytorch](https://pytorch.org/get-started/locally/).  
After activating the new environment, install Pytorch with your personal command.  

Example: install Pytorch in windows WSL2 with cuda 12.1  

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

`[your system specific Pytorch installation command]`

### Install MacaqueRetina
Navigate to the local MacaqueRetina git repo root and run: 

`pip install -e .`

### Optional: Install CxSystem2
If you want to run downstream simulations with CxSystem2, in terminal run:

`install_cxsystem.sh`
  - Resides in the same directory as this file

You may need to append execute to the access rights by command 
  
  `chmod ug+x install_cxsystem.sh`

## How to run

Start by activating your environment (see above)

The main files are in the project directory.

- project/project_conf_module.py
  - Primary interface to work with MacaqueRetina
  - You need to modify this file for your system and work each time you run the software
    - Comment in or out lines under `if __name__ == "main":` and run this file to operate the
    system during development
  - Before first run, update the model_root_path and git_repo_root_path to match your system
  - Take a moment to read the comments before starting

- project/project_manager_module.py
  - provides a facade for the rest of the code

After modifying the conf file, run always with the same command, e.g. from git repository root:  

`python project/project_manager_module.py`

## How to cite this project

## Contributing

Simo Vanni
Henri Hokkanen
