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

You can configure WSL2 by running the following commands in windows powershell

- `echo "[wsl2]" > ~/.wslconfig`
  - Writes the WSL2 configuration file to the user's home directory in Windows. Settings apply across all Linux distros running on WSL 2
  
- `echo "memory=26GB" >> ~/.wslconfig`
  - Limits VM memory to use no more than 26 GB, this can be set as whole numbers using GB or MB

- `echo "processors=12" >> ~/.wslconfig`
  - Sets the VM to use 12 virtual processors  

- `echo "swap=16GB" >> ~/.wslconfig`
  - Sets amount of swap storage space to 16GB  

**Make sure to apply WSL2 config changes by restarting WSL2 or your machine**.  
More examples and backgorund on WSL2 in [Microsoft pages](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

Next, follow the installation in linux, when you have the WSL2 up and you are in linux system.

## Install with Linux

Run the following commands to create and activate virtual environment.

- `python3.11 -m venv [your path]/ret_pt`
  - Make a new virtual environment named *ret_pt*
- `source [your root]/ret_pt2/bin/activate`
  - Activate the virtual environment

### Install pytorch

Pytorch is integrated in MacaqueRetina, because it enables GPU acceleration when CUDA device is avilable.  
Get your system- and cuda/cpu-specific [installation command for Pytorch](https://pytorch.org/get-started/locally/).  
After activating the new environment, install Pytorch with your personal command.  

Example: install Pytorch in windows WSL2 with cuda 12.1  
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- `[your system specific Pytorch installation command}`

### Install other dependencies

In terminal run:

- `install_ret_pt_linux.sh`
  - Resides in the same directory as this file
  - You may need to append execute to the access rights by command `chmod ug+x install_ret_pt_linux.sh`

## How to run

Start by activating your environment  
`source [your path]/ret_pt2/bin/activate`

The main files are in the project directory.

- project/project_conf_module.py
  - Primary interface to work with MacaqueRetina
  - You need to modify this file for your system and work each time you run the software

- project/project_manager_module.py
  - provides a facade for the rest of the code

After modifying the conf file, run always with the same command, e.g. from git repository root:  
`python project/project_manager_module.py`

## How to cite this project

## Contributing

Simo Vanni
Henri Hokkanen
