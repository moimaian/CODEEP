# -*- coding: utf-8 -*-
"""CODEEP User Manual content (en-US)."""
import os
FIG = os.environ.get("CODEEP_FIG", "fig")      # folder with the screenshots
def f(n): return os.path.join(FIG, n)


def conteudo(m):
    # ====================================================================== 1
    m.h1("1. Description")
    m.p("CODEEP is a computational platform for **de novo ligand generation** with deep learning, "
        "conditioned on the binding site of a target protein. Starting from the three-dimensional "
        "structure of the site, the program proposes candidate molecules, already with 3D coordinates, "
        "that complement the cavity. Its design covers the whole workflow: collection of "
        "protein–ligand complexes from public databases, validation and curation of the structures, "
        "exploration of the dataset, voxelization of the site into three-dimensional tensors, "
        "hyperparameter search, model training, evaluation and ligand generation for a new protein.")
    m.p("The model is a **conditional variational autoencoder (CVAE)**. The encoder is a 3D "
        "convolutional network that reads the site tensor — 32 physicochemical channels computed only "
        "from the atoms of the pocket itself — and summarizes it in a latent vector (256 dimensions by "
        "default). The decoder is a graph neural network that builds the ligand autoregressively, atom "
        "by atom: at each step it decides whether the molecule ends, from which already-placed atom it "
        "grows, the element and charge of the new atom, its 3D position and the bonds it makes. At "
        "generation time the latent vector is sampled and the decoder builds molecules for the given site.")
    m.p("The interface guides the user through this process without requiring programming knowledge. It "
        "was developed in Python 3.10 with the PyQt5/Qt 5.15 framework, in sequential tabs that match the "
        "steps of the workflow. The main integrated libraries are PyTorch, PyTorch Geometric, RDKit, "
        "BioPython, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn and Optuna, plus PySCF, CuPy "
        "(GPU voxelization) and the semiempirical program GFN-xTB (`xtb`), used to compute the electronic "
        "channels of the site, and P2Rank, used to search for binding pockets.")
    m.nota("In this manual, **base ligands** are the curated co-crystallized ligands that come with the "
           "training complexes: they serve as the answer (target) used to fit the decoder. The "
           "**reference ligand** is the ligand the user may optionally supply in STEP 6 to act as a "
           "template (*scaffold*) when evaluating the generated candidates. The two roles must not be "
           "confused: when generating for a new protein there is no base ligand.",
           "Terminology:")
    m.p("This manual describes version 1.1.0-1. The program is distributed under the GPL-3.0-or-later license.")

    # ====================================================================== 2
    m.h1("2. Installation")
    m.p("CODEEP runs on Linux. It is installed from the source code, as described below.")
    m.h2("2.1 Installing from source")
    m.p("Clone the repository into your home folder and run the program:")
    m.codigo("sudo apt-get install git python3 python3-venv   # if not already installed\n"
             "git clone https://github.com/moimaian/CODEEP.git $HOME/CODEEP\n"
             "cd $HOME/CODEEP && python3 ./CODEEP.py")
    m.p("Alternatively, download the compressed archive of the repository "
        "(`https://github.com/moimaian/CODEEP/archive/refs/heads/main.zip`), extract it into your home "
        "folder renaming it to `CODEEP`, and run `python3 ./CODEEP.py` from inside that folder.")
    m.h2("2.2 First run and virtual environment")
    m.p("On the first run the program creates the Python virtual environment at `$HOME/.venv/CODEEP`, where "
        "the dependencies are installed, and writes a shortcut in the application menu "
        "(`~/.local/share/applications/CODEEP.desktop`), from which it can then be launched. Next, the "
        "dependency installation window is shown, described in section 3.")
    m.p("To install packages manually from the terminal, the environment can be activated with:")
    m.codigo("source $HOME/.venv/CODEEP/bin/activate")

    # ====================================================================== 3
    m.h1("3. Prerequisites and dependency installation")
    m.p("The dependency installation window opens on first start-up and can be reopened at any time with the "
        "**Install Requirements** button on the HOME tab or through the **Help > Install Requirements** menu. "
        "It lists each package with its state (installed or missing), an editable version field and a "
        "description, installs the selected ones with `pip` in the CODEEP virtual environment and shows the "
        "progress in a log.")
    m.figura(f("dependencias.png"),
             "Dependency installation window, showing the state and version of each package.", 14.5)
    m.p("The PyTorch version is chosen according to the detected hardware: the CUDA 12 variant (`cu121`) when "
        "a compatible NVIDIA GPU is present, the CUDA 11 variant (`cu118`) for older installations and the "
        "CPU-only variant when there is no GPU. The packages fall into three groups:")
    m.tabela(["Group", "Packages"], [
        ["Essential", "PyQt5, PyTorch, torch-geometric, NumPy, Pandas, SciPy, Matplotlib, scikit-learn, "
                      "RDKit, BioPython"],
        ["Optional", "Seaborn, Optuna (needed only for STEP 3), PySCF, GPU4PySCF, CuPy and cuTENSOR "
                     "(GPU acceleration), tqdm, psutil, requests, openpyxl"],
        ["External tools", "`xtb` (GFN-xTB) and P2Rank 2.5.1, downloaded by the window itself into the "
                           "`tools/` folder; Java 17 or later (a system dependency required by P2Rank); "
                           "GROMACS and AmberTools, optional, only for the AMBER/CHARMM force fields of STEP 6"],
    ], [3.4, 12.6])
    m.p("Two items deserve attention:")
    m.bullets([
        "**GFN-xTB** is the recommended method for computing the donor/acceptor channels of the site in STEP 2. "
        "Without the `xtb` binary the program automatically falls back to a heuristic based on "
        "electronegativity, and records the method actually used in the `meta.json` file of each pair;",
        "A **GPU** is not mandatory, but it is practically required for training (STEP 4) at scale. "
        "Voxelization is also GPU-accelerated when CuPy is present; without it, NumPy on CPU is used."])
    m.p("The HOME tab shows the CPU, GPU and storage specifications of the machine and the versions "
        "actually detected for each package and tool (Software Specs group). This is the quickest way to "
        "check that the environment is complete.")
    m.figura(f("inicio.png"),
             "HOME tab: detected hardware and software specifications and summary of the pipeline steps.", 14.5)
    m.p("The **Tune parameters to this hardware** checkbox adapts to the detected equipment the fields that "
        "weigh most on performance: maximum Phase 1 entries, tensor precision, HOMO/LUMO channels method, "
        "maximum QM atoms, voxelization batch size, training batch size and learning rate, and device. When "
        "it is checked, each change is shown with its justification; when unchecked, the previous values "
        "are restored. The **View Terminal** button opens an integrated terminal.")

    # ====================================================================== 4
    m.h1("4. Directory organization")
    m.p("CODEEP works inside the folder where the `CODEEP.py` file is located, which is identified "
        "automatically. The Phase 1 data, shared among all projects, live in `data/`; each project has its "
        "own folder in `PROJECTS/`. The organization is as follows:")
    m.codigo("""CODEEP/
├── CODEEP.py                      main file
├── CODEEP_requirements.py         dependency check and installation
├── MODULES/                       pipeline modules (collection, curation, voxelization, training...)
├── ICONS/                         icons and logos
├── TUTORIALS/                     user manuals
├── tools/                         helper binaries (xtb, P2Rank)
├── data/                          Phase 1 data, shared among projects
│   ├── raw/                       downloaded structures (pdb/, sdf/)
│   ├── curated/<pair>/            receptor.pdb, ligand.pdb, binding_site.pdb
│   ├── tensors/<pair>/            site_tensor.npz, ligand_graph.pt, meta.json
│   ├── figures/                   exploration and evaluation figures
│   └── logs/                      logs of each step and diagnostics
├── checkpoints/phase1/            trained Phase 1 model
├── hpo/                           hyperparameter optimization results
└── PROJECTS/
    └── <date>_<name>/             one folder per project
        ├── job_config.json        project identification and search terms
        ├── step*.json             saved state of each tab
        ├── data/                  Phase 2 and 3 data (same structure)
        ├── checkpoints/           phase2/ and phase3/
        └── results/<target>/      generated candidates (STEP 6)""")
    m.p("The folder of each protein–ligand pair is named `<PDB code>_<ligand>_<chain><residue>`, for example "
        "`10gs_VWW_A210`. The tensors take about 1.5 MB per pair in float32 precision and about 0.45 MB in "
        "float16, values measured on the 32-channel set; the Hardware Specs — Storage group of the HOME tab "
        "estimates how many pairs fit on the available disk in each precision.")
    m.nota("The state of all tabs — parameters, paths and options — is saved automatically in the project "
           "folder and restored when it is reopened. The `step*.json` files keep the former internal numbering "
           "of the tabs: `step1.json` holds CONFIG; `step2.json`, `step2cur.json` and `step2exp.json`, "
           "STEP 1; `step3.json`, STEP 2; `step4.json`, STEP 3; `step5.json`, STEP 4; `step6.json`, "
           "STEP 5; and `step7.json`, STEP 6.")

    # ====================================================================== 5
    m.h1("5. General usage guidelines")
    m.p("The main window is organized in sequential tabs that follow the natural order of the workflow: "
        "**HOME**, **CONFIG** and **STEP 1 to STEP 6**. Navigation can be done through the tabs, through the "
        "**BACK** and **NEXT** buttons at the bottom of each tab, or through the **Menu** menu. The menu bar "
        "has three entries: **Menu** (Home, CONFIG, the six steps and Exit), **Help** (Install Requirements and "
        "Generate Final Report) and **About**.")
    m.p("In the upper right corner are the flags for switching between Portuguese and English — the choice is "
        "remembered between runs — and the CPU/GPU monitor button, which opens a window with processor, "
        "memory and graphics card usage during longer runs.")
    m.p("Almost every field has a help text, shown when the cursor is placed over the element. It gives the "
        "detailed description of the parameter, the recommended values and the consequences of each choice, "
        "and is the most complete and up-to-date reference on each control.")
    m.p("The parameters of each tab are organized in **collapsible groups**: clicking the title of a group "
        "(▾ open, ▸ collapsed) closes or opens it, which shortens the tab when you do not want to see every "
        "field. The run tabs (STEPS 1 to 6) share the same set of controls at the bottom:")
    m.bullets([
        "**Run** and **Stop**: processing runs in a process separate from the interface. Stop ends that "
        "process without closing CODEEP. While a STEP 1 group is running, the Run buttons of the other "
        "ones are disabled, because the three share a single log;",
        "**Execution Log**: a collapsible section with the real-time output of the processing. When it is "
        "opened, the window grows as much as needed so that the content does not overlap, and returns to "
        "the previous size when the section is closed;",
        "**Progress bar**: shows the percentage completed and the estimated remaining time."])
    m.p("Closing the window while a step is running is handled with a confirmation, to prevent accidentally "
        "interrupting a long training.")
    m.h2("5.1 The three data phases")
    m.p("Several tabs have the **Phase** field, which indicates which dataset the step acts on. The "
        "three-phase strategy makes it possible to start from a general model and specialize it:")
    m.bullets([
        "**Phase 1 — general pre-training.** Uses a broad external set of protein–ligand complexes "
        "(lp-PDBbind, RCSB, PLINDER or Binding MOAD). Its data are shared among all projects and live in "
        "`data/` and `checkpoints/phase1/`;",
        "**Phase 2 — specialization by family or organism.** Searches the RCSB by organism and by entity "
        "description, with the terms defined in the CONFIG tab. The data and the model live in the project "
        "folder, in `data/.../phase2` and `checkpoints/phase2`;",
        "**Phase 3 — specific targets.** Downloads structures from a list of PDB codes given in CONFIG. "
        "The data and the model live in `phase3`."])
    m.p("In training, Phase 2 starts, by default, from the Phase 1 model (fine-tuning) and Phase 3 starts from "
        "the Phase 2 model or, if it does not exist, from the Phase 1 one. In every case, the **Resume from** "
        "field lets you choose another starting point.")

    # ====================================================================== 6
    m.h1("6. CONFIG tab — project configuration")
    m.p("The CONFIG tab defines the working project. It is the first step: without a created or loaded "
        "project, the steps have nowhere to write their results.")
    m.figura(f("config.png"),
             "CONFIG tab: project identification, detected paths and search terms for Phases 2 and 3.", 14.5)
    m.h2("6.1 Project identification")
    m.p("The **Project Identification** group follows this order:")
    m.numerada([
        "**Date** — read-only, shows the current date and time;",
        "**New Project** — when checked, enables the **Job Name** field, where the name of the new project "
        "is typed;",
        "**Load Previous Project** — when checked, enables the **Previous Project** list, with the projects "
        "that already exist in `PROJECTS/`. The two options are mutually exclusive;",
        "**Job ID** — shows the name of the created or loaded project after the **Create / Load Project** "
        "button is pressed."])
    m.p("When a new project is created, the global Phase 1 structure (if it does not exist yet) and the project "
        "folder are created, named `<date>_<time>_<name>`, for example `2026-11-14_00-49_TESTE_ARCH5`, with the "
        "Phase 2 and 3 subdirectories. When an existing project is loaded, the saved state of all tabs is "
        "restored, including the training curves, when there is a history on disk.")
    m.h2("6.2 Detected paths")
    m.p("The **Detected Paths** group shows the state of the data and checkpoint folders, separated into "
        "**Global (shared — Phase 1)** and **Job-specific (Phase 2 / 3)**. For each folder, the ✔ or ✘ sign "
        "indicates whether it exists, followed by the count of files found (PDB, SDF, `.pt`, `.npz`). It is the "
        "quick way to see how far the pipeline has advanced in each phase.")
    m.h2("6.3 Search terms for Phases 2 and 3")
    m.p("The **Phase 2 / 3 — Target Search Terms** group gathers three text areas, one entry per line:")
    m.bullets([
        "**Organism terms** — used in the RCSB taxonomy search (Phase 2), for example `Homo sapiens`;",
        "**Entity description terms** — used in the RCSB keyword search (Phase 2), for example "
        "`protein kinase`;",
        "**Phase 3 — Specific PDB IDs** — PDB codes separated by commas or line breaks, for example "
        "`1ATP`, `2SRC`."])

    # ====================================================================== 7
    m.h1("7. STEP 1 — Data management")
    m.p("This step brings together, in a single tab, three independent blocks: **Data Collection**, **Data "
        "Validation and Curation** and **Data Exploration**. Each block has its own parameters and its own "
        "Run button, but the execution log and the progress bar are shared by the whole tab.")
    m.figura(f("etapa1.png"),
             "STEP 1 tab: data collection, validation/curation and exploration.", 14.5)
    m.h2("7.1 Data collection")
    m.p("Collection downloads the structures and ligands from the chosen databases. The controls of the "
        "group are:")
    m.bullets([
        "**Phase 1 Dataset** — chooses the external database: `lp-pdbbind` (about 17,000 curated pairs, with no "
        "data leakage; the default option), `rcsb-wide` (broad RCSB search for complexes with resolution up to "
        "2.5 Å), `plinder` (about 449,000 pairs, the largest database available) or `moad` (Binding MOAD, "
        "about 41,000 pairs, which requires prior registration on the database website);",
        "**Collection Phase** — Phase 1, Phase 2, Phase 3 or All Phases in sequence. Phase 2 uses the organism "
        "and description terms, and Phase 3 uses the PDB codes, both defined in CONFIG;",
        "**Max Phase 1 entries** — limits the number of downloaded entries (default 50,000);",
        "**MOAD Dir** — required only for the MOAD dataset, which must be downloaded beforehand;",
        "**Only show collected data status** — downloads nothing, only reports what already exists."])
    m.h2("7.2 Validation and curation")
    m.p("Curation reads the downloaded PDB files, validates the quality of the structures, extracts the "
        "binding site and produces the curated set used in the following steps. For each approved pair the "
        "following are written: `receptor.pdb` (protein without ligand and without water), `ligand.pdb` (the "
        "base ligand, at the crystal position), `binding_site.pdb` (site residues) and `pair_info.json`. "
        "The parameters are:")
    m.tabela(["Field", "Default", "Function"], [
        ["Phase", "1", "Dataset to curate."],
        ["Workers (0=auto)", "0", "Number of parallel processes; 0 detects automatically."],
        ["Site cutoff (Å)", "6.0", "Maximum distance between a residue and the ligand for the residue to be "
                                   "part of the site."],
        ["Min. site residues", "5", "Structures whose site has fewer residues than this are rejected."],
        ["Max. ligand MW (Da)", "800", "Heavier ligands are rejected."],
        ["Min. heavy atoms", "7", "Smaller ligands are rejected."],
        ["ThreadPoolExecutor / ProcessPoolExecutor", "Processes", "Type of parallelism. Processes is the "
                                                                  "default, because reading PDB files with "
                                                                  "BioPython does not benefit from threads."],
    ], [4.8, 2.2, 9.0])
    m.p("Besides these filters, curation automatically removes solvents, ions, buffers, crystallization "
        "additives, polyethylene glycols, detergents and lipids, and rejects ligands covalently bound to the "
        "protein (such as modified residues and glycosylations), since they are not ligands in the sense of the "
        "problem. Nucleotides and cofactors are also excluded by default: they concentrate the element "
        "phosphorus and the purine core and would bias the elemental composition of the model.")
    m.nota("Turning off the cofactor exclusion is recommended only when the target is, precisely, a "
           "nucleotide-dependent enzyme. This option is in `CURATE_CONFIG` in the `pipeline_curate.py` module.")
    m.h2("7.3 Data exploration")
    m.p("The **View Dashboard** button generates, for the chosen phase, the exploratory analysis dashboard of the "
        "curated set and shows it in a window. The dashboard gathers the distributions of molecular weight, "
        "ligand size and binding-site size, the most frequent ligands and summary statistics, and is the "
        "recommended way to check the set before voxelization. The figures are saved in `data/figures/`, in "
        "PNG and SVG.")
    m.figura(f("summary_dashboard.png"),
             "Exploration dashboard of the curated set (View Dashboard).", 14.0)

    # ====================================================================== 8
    m.h1("8. STEP 2 — Voxelization")
    m.p("This step converts each curated pair into data that the neural network consumes: a **3D tensor of the "
        "site** and a **molecular graph of the base ligand**. For each pair, the following files are written to "
        "`data/tensors/<pair>/`: `site_tensor.npz` (compressed tensor), `ligand_graph.pt` (graph in the PyTorch "
        "Geometric format, with atom types, charges, bonds and the 3D coordinates of the crystal) and "
        "`meta.json`.")
    m.figura(f("etapa2.png"),
             "STEP 2 tab: scope, site preparation, voxel grid and quantum channels.", 14.5)
    m.p("The site tensor has **32 channels**, all computed exclusively from the atoms of the binding pocket "
        "itself and never from the ligand. This guarantees that the tensor is identical in training and in "
        "generation, when there is no ligand at all. The channels form three groups:")
    m.tabela(["Group", "Channels", "Content"], [
        ["Element", "5", "Gaussian density separated by species: C, N, O, S and coordinating metals."],
        ["Physicochemical", "6", "Hydrophobicity (Crippen), aromaticity, partial charge (Gasteiger), "
                                 "electrostatic potential (Coulomb) and the electron donor and acceptor "
                                 "fields (HOMO- and LUMO-like)."],
        ["Amino acid", "21", "Residue identity, one channel for each of the 20 canonical amino acids and one "
                             "for the others."],
    ], [3.2, 1.8, 11.0])
    m.h2("8.1 Scope and execution")
    m.bullets([
        "**Phase** — set to voxelize (1, 2 or 3);",
        "**Specific pair** — processes only the given pair; empty processes all of them;",
        "**Workers** and **Parallelize pairs** — by default the pairs are processed one at a time, the "
        "validated behavior. Checking Parallelize pairs processes several at once, splitting the threads "
        "among them; the expected gain is modest, 15 to 25%, because the quantum calculation, the slowest "
        "part, already uses almost every core on each pair;",
        "**Vox batch size** — atoms per batch in the Gaussian voxelization (default 500; 500 is recommended "
        "for a GPU with 8 to 16 GB and 100 for CPU). Larger values reduce the per-batch overhead but "
        "increase memory use;",
        "**Tensor precision** — precision with which the tensor is written to disk, `float32` (default) or "
        "`float16`. It changes neither the calculation nor the training; `float16` reduces disk space to "
        "about a third, at the cost of a small rounding error."])
    m.h2("8.2 Site preparation")
    m.bullets([
        "**Protonate site residues (pH-based)** — adds explicit hydrogens to the ionizable residues (Asp, Glu, "
        "His, Cys, Tyr, Lys and Arg) according to the tabulated pKa and the chosen pH, and starts using "
        "per-atom Gasteiger charges. Turning it off returns to the legacy behavior, without hydrogens;",
        "**Site pH** — pH used to decide the protonation state (default 7.4)."])
    m.h2("8.3 Voxel grid and channels")
    m.bullets([
        "**Grid size (N³)** — number of voxels per side (default 32, that is, 32 × 32 × 32);",
        "**Padding (Å)** — margin added around the site when defining the box (default 3.0 Å). The voxel size "
        "results from the largest dimension of the site, plus the padding, divided by the grid;",
        "**Use ECFP4 in graph** — adds the ECFP4 fingerprint to the ligand graph. It reduces diversity and "
        "comes unchecked."])
    m.h2("8.4 Quantum channels (QM)")
    m.bullets([
        "**Site HOMO/LUMO channels method** — level of theory for the donor/acceptor field: "
        "**GFN-xTB per residue, parallel** (recommended, requires the `xtb` binary and takes about 0.25 s per "
        "pair), **RHF/STO-3G per residue** (fragmented quantum calculation, slower) or **Heuristic** "
        "(electronegativity, no quantum calculation);",
        "**Max QM atoms** — limit of heavy atoms for the full calculation (default 60);",
        "**QM basis** — basis of the per-residue RHF method (`sto-3g`, `3-21g` or `6-31g`); it is ignored by "
        "GFN-xTB and by the heuristic."])
    m.nota("The chosen method and basis are saved in the trained checkpoint. In STEP 6, generation "
           "automatically uses the voxelization configuration of the training, so that the tensor of the new "
           "site has the same distribution as the training set.")

    # ====================================================================== 9
    m.h1("9. STEP 3 — Hyperparameter optimization")
    m.p("This step is optional. It uses Optuna to search for hyperparameter combinations on a cheap subset of "
        "the data and in few epochs per attempt (*trial*). Since the ranking comes from a subset and/or from "
        "early epochs, the search is a **proxy**: use it to choose two or three configurations and confirm "
        "them with a full training in STEP 4.")
    m.figura(f("etapa3.png"),
             "STEP 3 tab: search budget and hyperparameter search space.", 14.5)
    m.h2("9.1 Search budget")
    m.bullets([
        "**Search algorithm** — defines which configuration to try next: TPE (Optuna, default), Random "
        "Search, Grid Search or Bayesian (Gaussian Process). In Grid Search, *Trials* becomes the cap on "
        "the grid;",
        "**Phase (dataset)**, **Trials** (default 40), **Epochs per trial** (default 15) and **Dataset "
        "fraction** (default 0.20);",
        "**ASHA min. epochs** and **ASHA reduction factor** — configure Successive Halving, which is a "
        "*pruning* algorithm: it decides when to stop a trial in progress and combines with any search "
        "algorithm;",
        "**Seed**, **Device** and **Study name** — reproducibility, GPU or CPU, and the name of the Optuna "
        "study;",
        "**Resume from** — `best_model.pt` of a previous training, to tune fine-tuning in Phases 2 and 3; "
        "**Freeze encoder** — keeps the encoder fixed in that fine-tuning;",
        "**Keep checkpoints from every trial** — uses much more disk;",
        "**Output folder** — default `hpo/`."])
    m.h2("9.2 Search space")
    m.p("Each hyperparameter has a checkbox and a range: `min,max` for continuous or integer parameters and a "
        "list of values for categorical ones. The range is editable only when the box is checked. The "
        "parameters are divided into three columns by priority; the high- and medium-priority ones come "
        "pre-selected:")
    m.tabela(["Priority", "Hyperparameters"], [
        ["High", "`lr`, `beta_max`, `pos_weight`, `stop_weight`"],
        ["Medium", "`beta_warmup`, `dropout`, `hidden_dim`, `n_gnn_layers`"],
        ["Low", "`latent_dim`, `batch_size`, `weight_decay`, `grad_clip`"],
    ], [3.0, 13.0])
    m.p("The **Objective** can be `val_recon` (reconstruction only, invariant to beta) or `val_total` "
        "(reconstruction plus beta·KL). Since `val_total` is valid only with a fixed beta, it is disabled "
        "automatically while `beta_max` or `beta_warmup` are in the search space. The **Select all**, "
        "**Restore defaults** and **Clear all** buttons act on the checkboxes, and **Estimate disk usage** "
        "computes how much the study will take. Note: `batch_size` interacts strongly with `lr`; searching "
        "it in isolation is misleading, and it only makes sense with `lr` also checked.")
    m.p("At the end, the output folder contains `best_params.json` (best combination), `hpo_trials.csv` (all "
        "trials) and `optuna_study.db` (the study, which can be resumed).")

    # ====================================================================== 10
    m.h1("10. STEP 4 — Training")
    m.p("This step trains the CVAE on the voxelized pairs. The encoder receives the site tensor and produces the "
        "distribution of the latent vector; the decoder, under *teacher forcing*, receives the latent vector "
        "and the partial graph of the base ligand and learns to predict the next construction step. Training "
        "optimizes the reconstruction loss and the Kullback–Leibler (KL) divergence term, weighted by a beta "
        "that grows gradually (*warmup*).")
    m.figura(f("etapa4.png"),
             "STEP 4 tab: training parameters, organized in collapsible groups.", 14.5)
    m.p("The split into training, validation and test sets (80%, 10% and 10%) is done by **PDB code**: all the "
        "ligands of the same structure stay in the same partition. No separation by sequence similarity "
        "between different proteins is made.")
    m.h2("10.1 Parameters")
    m.tabela(["Group", "Field", "Default", "Function"], [
        ["Capacity", "Latent dim", "256", "Dimension of the latent vector."],
        ["", "Hidden dim", "256", "Width of the GNN decoder."],
        ["", "GNN layers", "4", "Depth of the decoder."],
        ["Optimization", "Epochs", "150", "Maximum number of epochs."],
        ["", "Batch size", "32", "Pairs per step."],
        ["", "Learning rate", "4e-4", "AdamW optimizer rate."],
        ["", "Patience", "15", "Epochs without validation improvement before stopping (*early stopping*)."],
        ["Regularization", "Dropout", "0.20", "Model regularization."],
        ["", "Weight decay", "1e-4", "AdamW L2 regularization."],
        ["", "Gradient clipping", "0.50", "Maximum gradient norm per step; protects against gradient "
                                         "explosion."],
        ["Loss weights and KL", "Beta KL (max)", "0.5", "Maximum weight of the KL term; it is the main "
                                                        "cause of posterior collapse."],
        ["", "Beta warmup", "30", "Epochs until beta reaches the maximum."],
        ["", "Position weight", "2.0", "Weight of the 3D position term. It is a log-likelihood, not an error "
                                      "in ångström."],
        ["", "Stop weight", "1.0", "Controls the size of the molecules: too low yields ligands that never "
                                   "stop growing, too high yields fragments."],
        ["", "Focal weight", "1.0", "Weight of the choice of the atom from which the molecule grows."],
        ["Scheduled sampling", "Scheduled sampling", "0.25", "Maximum fraction of atoms of the partial graph "
                                                              "shown to the decoder with the element it "
                                                              "itself predicted; 0 turns it off."],
        ["", "Sched. sampling start", "20", "Epochs of pure *teacher forcing* before the substitution starts."],
        ["", "Sched. sampling ramp", "40", "Epochs to go from 0 to the maximum."],
        ["Execution", "Phase", "1", "Training set."],
        ["", "Device", "cuda", "GPU or CPU, from the detected GPUs."],
        ["Checkpoints", "Resume from", "—", "`best_model.pt` to start from (fine-tuning)."],
        ["", "Save ckpt to", "checkpoints/", "Output folder."],
        ["", "Freeze encoder", "unchecked", "Keeps the encoder fixed (Phase 3 fine-tuning)."],
    ], [3.3, 3.6, 2.2, 6.9])
    m.h2("10.2 Monitoring")
    m.p("The **Training Curves** section, collapsed when the tab opens, shows the training and validation loss "
        "in real time at each epoch; when expanded, the window grows to accommodate it. The **Execution Log** "
        "shows, per epoch, the losses and the diagnostic metrics, among them the mean geometric error of the "
        "positions (`posMAE`, in Å).")
    m.p("At the end of training, the checkpoint folder contains:")
    m.bullets([
        "`best_model.pt` — the model with the lowest validation loss, which also carries the configuration "
        "used in training (architecture, position frame and voxelization parameters);",
        "`epoch_NNNN.pt` — weights saved every 10 epochs;",
        "`training_history.csv` and `training.log` — per-epoch history and the complete record of the training."])
    m.nota("A drop of the KL to values near zero indicates *posterior collapse*: the decoder starts ignoring "
           "the latent vector and generation loses diversity. The active units diagnostic, in STEP 5, helps "
           "detect this situation.")

    # ====================================================================== 11
    m.h1("11. STEP 5 — Evaluation")
    m.p("This step evaluates the trained model. The **Run** button runs, in sequence, the analysis of the "
        "training history (epochs trained, best epoch, gap between training and validation, final KL and "
        "detected problems), the evaluation on the test set, the diagnostic of the latent space and, "
        "optionally, the generation of example ligands.")
    m.figura(f("etapa5.png"),
             "STEP 5 tab: model and data, and the three buttons for figures and diagnostics.", 14.5)
    m.h2("11.1 Model and data")
    m.bullets([
        "**Phase** — phase of the model to evaluate;",
        "**Checkpoints** — model folder; when empty, uses `checkpoints/phase<N>/` of the chosen phase;",
        "**Generate ligands during evaluation** — besides the metrics, generates a few ligands to compute "
        "validity, uniqueness and chemical properties."])
    m.h2("11.2 Figures and diagnostics")
    m.p("The **Figures and Diagnostics** group has three buttons on the same line, in the order in which they "
        "are usually used:")
    m.bullets([
        "**Training Curves** — shows the `training_curves` figure, with the evolution of the losses over the "
        "epochs;",
        "**Performance Report** — shows the `performance_report` figure, with the test loss by component, the "
        "QED distribution of the generated ligands and a table of generation metrics;",
        "**Active Units Diagnostic** — runs the diagnostic described in section 11.3."])
    m.p("The figures are saved in PNG and SVG in `data/figures/` (Phase 1) or in the figures folder of the "
        "project (Phases 2 and 3). They exist only after the evaluation has been run.")
    m.figura(f("training_curves.png"), "Training curves of the Phase 1 model.", 14.0)
    m.figura(f("performance_report.png"),
             "Performance report: test-set losses, QED distribution and generation metrics.", 16.0)
    m.h2("11.3 Active Units Diagnostic")
    m.p("It checks how many of the latent space dimensions the encoder actually uses, through the *active "
        "units* metric (Burda, Grosse and Salakhutdinov, 2016). For each dimension, the variance of the "
        "posterior mean across the sites of the validation set is computed; the dimension is considered "
        "active when this variance exceeds 0.01. The diagnostic detects **partial** posterior collapse, which "
        "a mean KL value alone can hide: a moderate mean KL can still mean few active dimensions and the "
        "rest dead.")
    m.p("The result appears in a text window, with the checkpoint, the saved epoch, the set and the number of "
        "samples used, the threshold, the total of active units and an interpretation: good utilization (50% "
        "or more active), partial utilization (10% to 50%) or strong sign of partial collapse (less than 10%).")
    m.figura(f("active_units.png"),
             "Active Units Diagnostic window, showing the path of the saved log.", 12.0)
    m.p("Each run is also **saved to disk**, in the log folder of the chosen phase (`data/logs/` in Phase 1 and "
        "`<project>/data/logs/phase<N>/` in Phases 2 and 3), for later consultation and use in reports:")
    m.bullets([
        "`active_units.log` — text report; each run is appended at the end, with date and time;",
        "`active_units.json` — the most recent run in structured format, with checkpoint, epoch, set, samples, "
        "threshold, active units, total, fraction and the variance of each dimension."])
    m.p("The **Open log** button of the window opens the file in the system's default application.")

    # ====================================================================== 12
    m.h1("12. STEP 6 — Ligand candidate generation")
    m.p("This step generates candidate ligands for a protein the model did not see in training. From the PDB "
        "file, the program extracts the binding site, computes the 32-channel tensor, samples the latent "
        "vector, builds the molecules, minimizes the geometry, checks for clashes with the protein, ranks the "
        "candidates and saves the results.")
    m.figura(f("etapa6.png"),
             "STEP 6 tab: target and output, site definition, sampling, channels, filters and final geometry.", 14.5)
    m.h2("12.1 Target, model and output")
    m.bullets([
        "**Model phase** and **Checkpoints** — model to use; when empty, uses `checkpoints/phase<N>/`. The "
        "`best_model.pt` file must exist;",
        "**Protein (PDB)** — PDB file of the target protein. It is the only required field;",
        "**Site residues (PDB)** — optional: a PDB containing only the site residues, when the user already "
        "knows exactly which they are;",
        "**Ref. ligand** — optional: the **reference ligand**, in SDF. When supplied, the Tanimoto "
        "similarity of each candidate to it enters the score and the results table;",
        "**Output** — folder of the results; when empty, uses `PROJECTS/<project>/results/<target>/`."])
    m.h2("12.2 Site definition")
    m.p("The site can be defined in three mutually exclusive ways:")
    m.numerada([
        "**By the co-crystallized ligand** — if the protein has a ligand (`HETATM`), the residues within 6 Å of "
        "it define the site;",
        "**By center coordinates** — check **Define site by center coordinates**, enter X, Y and Z (in the same "
        "frame as the PDB) and the **Site radius** (default 10 Å). The residues with at least one atom "
        "within the radius define the site. Use it when the protein has no ligand;",
        "**By pocket search** — check **Heuristic search (P2Rank)**. P2Rank predicts the pockets and the one "
        "chosen in **Pocket to use** (1 = highest score) defines the site. It requires P2Rank and Java 17 or "
        "later; the full ranking is saved in `pockets_p2rank.json`."])
    m.p("Checking the heuristic search disables the manual coordinates. These two options cannot be combined "
        "with a site-residues PDB: the program warns about the conflict and does not run.")
    m.h2("12.3 Sampling")
    m.bullets([
        "**No. samples** (default 200) — how many molecules to generate;",
        "**Temperature** (default 1.0) — diversity of the sampling. High values also flatten the distribution "
        "of atom types, and generation tends to over-represent elements such as P, F and S relative to "
        "training; reducing it to about 0.7 brings the elemental composition closer to the database;",
        "**Stop threshold** (default 0.5) — confidence the model must reach to end the molecule. Raise it "
        "(0.6 to 0.7) for larger ligands and lower it for smaller ones; unlike the stop weight, it does not "
        "require retraining;",
        "**Bond threshold** (default 0.5) — minimum probability to accept a bond beyond the anchor with the "
        "focal atom. Lowering it favors ring closure, at the risk of spurious bonds;",
        "**Site cutoff (Å)** (default 6.0) — distance used to cut out the site residues when there is a "
        "ligand."])
    m.p("During sampling, a **steric guard** rejects and redraws any atom that clashes with the site atoms, "
        "and also atoms placed at distances incompatible with the neighboring bonds. This markedly reduces "
        "clashes of the molecule with the protein, with no need to retrain the model.")
    m.h2("12.4 Site channels")
    m.p("The **Site Channels (must match training)** group repeats the protonation, pH, HOMO/LUMO channels "
        "method and QM basis options of STEP 2. When the chosen checkpoint carries the voxelization "
        "configuration saved at training, it always prevails over these fields, with a warning in the log if "
        "they differ; they only take effect for old checkpoints that lack this configuration.")
    m.h2("12.5 Filters and ranking")
    m.bullets([
        "**Min. site atoms** (default 5) — minimum number of heavy atoms of the site to accept the target;",
        "**Min. heavy atoms** (default 7) — discards candidates with fewer heavy atoms than this, after "
        "ranking; thus, Top N means the N best that pass the filter. Use 1 to turn it off;",
        "**Top N report** (default 20) — how many candidates enter the output files."])
    m.p("Each candidate receives a composite score between 0 and 1, the sum of: 40% QED, 20% synthetic "
        "accessibility (SA score), 15% the Lipinski rule, 10% the Veber rule, 5% joint compliance with limits "
        "of molecular weight, logP, TPSA and rotatable bonds, and 10% Tanimoto similarity to the reference "
        "ligand, when supplied. Candidates that clash with the site have the score multiplied by a factor "
        "that decreases with the number of clashes, so that a high QED does not compensate for an impossible fit.")
    m.h2("12.6 Final geometry")
    m.bullets([
        "**Final geometry FF** — force field of the final minimization of the **complex** (ligand plus site), "
        "done before the clash check: **MMFF/UFF** (RDKit, default, no extra dependency; UFF covers metals "
        "such as Zn, Fe and Mg), **AMBER ff14SB** or **CHARMM36** (require GROMACS and AmberTools; without "
        "them the program falls back to MMFF/UFF with a warning in the log) or **None** (raw decoder "
        "geometry, for diagnostics);",
        "**Induced-fit (flexible side chains)** — unchecked, the site is rigid and only the ligand moves; "
        "checked, only the backbone (N, Cα, C, O) of the site residues is locked and the side chains adjust to "
        "the ligand during the minimization."])
    m.h2("12.7 Results")
    m.p("At the end, the output folder contains:")
    m.tabela(["File", "Content"], [
        ["`candidates.csv`", "Table with all candidates and their metrics: SMILES, score, QED, molecular "
                             "weight, logP, hydrogen-bond donors and acceptors, TPSA, rotatable bonds, rings, "
                             "Lipinski and Veber rules, SA score, number of atoms, Tanimoto to the reference "
                             "ligand, presence and number of clashes with the site."],
        ["`top<N>_candidates.sdf`", "3D structures of the N best candidates."],
        ["`top<N>_candidates.png`", "Grid with the 2D structures of the N best candidates."],
        ["`report.html`", "Full interactive report."],
        ["`site_tensor.npz`", "Site tensor used in generation."],
        ["`site_residues_used.pdb`", "Site residues actually used."],
        ["`pockets_p2rank.json`", "P2Rank pocket ranking, when the heuristic search was used."],
    ], [4.6, 11.4])
    m.figura(f("top20.png"), "2D grid of the best generated candidates (`top20_candidates.png`).", 11.5)
    m.nota("The candidates are starting-point proposals generated by a statistical model. The score and the "
           "metrics guide prioritization but do not replace validation by molecular docking, molecular "
           "dynamics and, ultimately, experimental assays.", "Important:")

    # ====================================================================== 13
    m.h1("13. Generate Final Report")
    m.p("The **Help > Generate Final Report** option saves the state of all tabs of the current project to the "
        "`final_report.json` file, in the project folder, together with the program version and the date of "
        "generation. This file is the data source of the final project report, which is still under "
        "development; for now, it records the parameters used in each step and is useful for documenting and "
        "reproducing an analysis.")

    # ====================================================================== 14
    m.h1("14. Recommended workflow")
    m.p("The sequence of tabs matches the recommended workflow. A typical outline, starting from an external "
        "set and specializing the model, is:")
    m.numerada([
        "**HOME** — check the specifications and install the missing dependencies;",
        "**CONFIG** — create the project and, for Phases 2 and 3, fill in the search terms;",
        "**STEP 1** — collect the data, curate it and examine the exploration dashboard;",
        "**STEP 2** — voxelize the curated pairs (GFN-xTB, if available);",
        "**STEP 3** (optional) — search for hyperparameters and choose the best configurations;",
        "**STEP 4** — train the model, in Phase 1 and then, if applicable, in Phases 2 and 3 by fine-tuning;",
        "**STEP 5** — evaluate the model, check the curves and run the Active Units diagnostic;",
        "**STEP 6** — generate candidates for the target protein, defining the site by ligand, coordinates or "
        "P2Rank, and analyze the results;",
        "**Help > Generate Final Report** — record the parameters of the project."])
    m.nota("If the goal is only to generate ligands for a new protein with an already trained model, STEPS 1 to "
           "5 can be skipped: just point to the checkpoint in STEP 6.")

    # ====================================================================== 15
    m.h1("15. License, repository and citation")
    m.p("CODEEP is distributed under the GNU General Public License, version 3 or later "
        "(GPL-3.0-or-later). The source code and the tutorials are available at "
        "`https://github.com/moimaian/CODEEP`.")
    m.p("The program uses third-party components, each with its own license, among them RDKit, PyTorch, PyTorch "
        "Geometric, GFN-xTB and P2Rank. When publishing results obtained with CODEEP, cite the program and its "
        "version, shown in the footer of the HOME tab and in the About window. For P2Rank, cite Krivák and "
        "Hoksza (2018); for the active units metric, Burda, Grosse and Salakhutdinov (2016).")
