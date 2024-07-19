from setuptools import setup, find_packages
import re


# Function to read the version from __init__.py
def get_version():
    with open("project/__init__.py", "r") as fh:
        for line in fh:
            match = re.match(r"^__version__ = ['\"]([^'\"]*)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Version information not found.")


# Reading long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="MacaqueRetina",
    version=get_version(),
    author="Simo Vanni, Henri Hokkanen",
    author_email="simo.vanni@helsinki.fi",  # Replace with actual email
    description="Python software to build a model of macaque retina and convert visual stimuli to cortical input",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VisualNeuroscience-UH/MacaqueRetina",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "Pillow",
        "imageio",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
    ],
    scripts=["install_ret_pt_linux.sh"],
    entry_points={
        "console_scripts": [
            "run_macaqueretina=project.project_manager_module:main",
        ],
    },
)
