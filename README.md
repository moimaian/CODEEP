# CODEEP

CODEEP is a computational platform for **de novo ligand generation** with deep learning, conditioned on the binding site of a target protein. From the three-dimensional structure of the site, it proposes candidate molecules, with 3D coordinates, that complement the cavity. Its design covers the whole workflow: collection of protein–ligand complexes from public databases (lp-PDBbind, RCSB, PLINDER, Binding MOAD), validation and curation of the structures, voxelization of the binding site into 3D tensors with 32 physicochemical channels, hyperparameter search, training of a conditional variational autoencoder (3D-CNN site encoder and autoregressive graph-neural-network ligand decoder), evaluation, and generation of ranked ligand candidates for a new protein.

The interface guides the user through this process, reducing the need for programming knowledge. It was developed in Python 3.10 with the PyQt5/Qt 5.15 framework, organized into sequential tabs (HOME, CONFIG and STEP 1 to STEP 6). The main libraries integrated are PyTorch, PyTorch Geometric, RDKit, BioPython, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn and Optuna, plus PySCF, CuPy, the semiempirical program GFN-xTB (xtb) and P2Rank.

## Installation

### Option 1 — Debian/Ubuntu package (.deb)

```bash
sudo apt install ./codeep_<version>_all.deb
```

The package installs the program in `/opt/codeep`, the launcher `/usr/bin/codeep` and a menu entry. On the first launch from the menu, the launcher copies the program to `$HOME/CODEEP` (never touching your data, models and projects) and a terminal window asks for confirmation to create the Python virtual environment. See `packaging/deb/README.md` for details and `packaging/deb/build.sh` to build the package.

### Option 2 — From source

**Download the compressed file from the link:**
https://github.com/moimaian/CODEEP/archive/refs/heads/main.zip

**Extract it into your home folder, creating `$HOME/CODEEP` (remove `-main` from the name), and run it:**

```bash
sudo apt-get install unzip      # if not already installed
unzip CODEEP-main.zip
mv $HOME/Downloads/CODEEP-main $HOME/CODEEP
cd $HOME/CODEEP && python3 ./CODEEP.py
```

(or `git clone https://github.com/moimaian/CODEEP.git $HOME/CODEEP`). If you do not have Python 3 on your system:

```bash
sudo apt update
sudo apt install python3 python3-venv -y
```

On the first run, the Python virtual environment is created at `$HOME/.venv/CODEEP`, where the dependencies are installed. A window then lists the packages that are missing, which can be installed from the interface itself (HOME tab > *Install Requirements*, or *Help > Install Requirements*). To activate the environment in a terminal:

```bash
source $HOME/.venv/CODEEP/bin/activate
```

## Prerequisites

All prerequisites are installed from the *Install Requirements* window. The PyTorch build is chosen automatically from the detected hardware (CUDA 12, CUDA 11 or CPU). Besides the Python packages, the window also downloads the helper tools `xtb` (GFN-xTB) and P2Rank into `$HOME/CODEEP/tools`. Two items are system dependencies:

- **Java 17 or later**, required by P2Rank (pocket search);
- **GROMACS and AmberTools**, optional, only for the AMBER ff14SB / CHARMM36 force fields of the final geometry in STEP 6.

An NVIDIA GPU is not mandatory, but it is practically required for training at scale.

This application was developed for Linux, targeting Debian/Ubuntu and their flavors; other distributions are not guaranteed to work.

## More information

See the User Manual in the `TUTORIALS` folder (Portuguese: `manual_do_usuario_CODEEP_pt_BR`, English: `User_manual_CODEEP_en-US`).

## License

CODEEP is free software and is distributed under the terms of the GNU General Public License v3.0 or later (GPL-3.0-or-later).

Copyright (C) 2024–2026 Moisés Maia.

See the `LICENSE.txt` file for details.
