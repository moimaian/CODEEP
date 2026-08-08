# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# CODEEP – Deep Learning Pipeline for Ligand Generation
# Part of the CODOC · CODYN · CODEEP · CODRUG suite
# Copyright (C) 2024–2026 Moisés Maia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""Interface texts in English / Brazilian Portuguese for CODEEP.

Mirrors the pattern used by AgendaLab/CODRUG's i18n.py: a flat dict of {key: {"en": ..., "pt": ...}}
plus a single t(key, idioma, **kwargs) lookup. CODEEP's UI was originally written entirely in
English, so "en" values are always the original, unmodified text (switching to English can never
change existing behavior/wording) and "pt" values are the added translation — with one exception:
a handful of tooltips added in a later session (site protonation/pH, final-geometry FF,
induced-fit) were originally written in Portuguese; for those, "pt" preserves that original
wording and "en" is the newly-added translation, so the *meaning* stays consistent even though
which language is "original" flips for that small subset.

t() falls back to returning the key itself when a key has no entry yet (or no entry for the
requested language) - this lets the interface be translated incrementally without ever crashing
or showing a blank string for text that hasn't been ported to i18n.t(...) yet.

Scope: this module covers the CODEEP.py GUI chrome (window/menu/tabs, group titles, field labels,
buttons, checkboxes, tooltips, dialogs). It does NOT cover the pipeline's own log/print output
(MODULES/pipeline_*.py, run in a child process) — that text is not part of the "interface" this
toggle controls, the same scope boundary CODRUG's i18n.py uses.
"""

IDIOMA_PADRAO = "en"

_TEXTOS = {
    # ---------------------------------------------------------------- Main window chrome
    "app_titulo": {
        "en": "CODEEP – Deep Learning Ligand Generation Pipeline",
        "pt": "CODEEP – Pipeline de Geração de Ligantes por Aprendizado Profundo",
    },
    "tooltip_bandeira_pt": {"en": "Português", "pt": "Português"},
    "tooltip_bandeira_en": {"en": "English", "pt": "English"},

    # ---------------------------------------------------------------- Menu
    "menu_title": {"en": "Menu", "pt": "Menu"},
    "menu_home": {"en": "Home", "pt": "Início"},
    "menu_step1": {"en": "Step 1 – Job", "pt": "Etapa 1 – Job"},
    "menu_step2": {"en": "Step 2 – Collection", "pt": "Etapa 2 – Coleta"},
    "menu_step3": {"en": "Step 3 – Curation", "pt": "Etapa 3 – Curadoria"},
    "menu_step4": {"en": "Step 4 – Exploration", "pt": "Etapa 4 – Exploração"},
    "menu_step5": {"en": "Step 5 – Voxelization", "pt": "Etapa 5 – Voxelização"},
    "menu_step6": {"en": "Step 6 – Training", "pt": "Etapa 6 – Treinamento"},
    "menu_step7": {"en": "Step 7 – Evaluation", "pt": "Etapa 7 – Avaliação"},
    "menu_step8": {"en": "Step 8 – Generation", "pt": "Etapa 8 – Geração"},
    "menu_exit": {"en": "Exit", "pt": "Sair"},
    "menu_help": {"en": "Help", "pt": "Ajuda"},
    "menu_install_requirements": {"en": "Install Requirements", "pt": "Instalar Dependências"},
    "menu_about": {"en": "About", "pt": "Sobre"},

    # ---------------------------------------------------------------- Tab names (short)
    "tab_home": {"en": "Home", "pt": "Início"},
    "tab_step1": {"en": "Step 1", "pt": "Etapa 1"},
    "tab_step2": {"en": "Step 2", "pt": "Etapa 2"},
    "tab_step3": {"en": "Step 3", "pt": "Etapa 3"},
    "tab_step4": {"en": "Step 4", "pt": "Etapa 4"},
    "tab_step5": {"en": "Step 5", "pt": "Etapa 5"},
    "tab_step6": {"en": "Step 6", "pt": "Etapa 6"},
    "tab_step7": {"en": "Step 7", "pt": "Etapa 7"},
    "tab_step8": {"en": "Step 8", "pt": "Etapa 8"},

    # ---------------------------------------------------------------- HW monitor (top-right corner)
    # Monitor num PROCESSO separado (ver MODULES/hw_monitor_window.py), mesmo padrão do CODRUG —
    # só um botão aqui na janela principal; os textos ao vivo (CPU/GPU/temperatura) ficam dentro
    # do próprio hw_monitor_window.py, fora do escopo deste toggle (janela/processo distinto).
    "btn_cpu_gpu_monitor": {"en": "CPU/GPU Monitor", "pt": "Monitor de CPU/GPU"},
    "tooltip_cpu_gpu_monitor": {
        "en": "Open a separate CPU/GPU monitor window that keeps updating even while the main "
              "window is busy running a heavy task (e.g. Step 6 Training or Step 8 Generation).",
        "pt": "Abre uma janela separada de monitor de CPU/GPU que continua atualizando mesmo "
              "quando a janela principal estiver ocupada rodando uma tarefa pesada (ex.: Etapa 6 "
              "Treinamento ou Etapa 8 Geração).",
    },
    "msg_title_monitor": {"en": "Monitor", "pt": "Monitor"},
    "msg_title_monitor_error": {"en": "Monitor error", "pt": "Erro no monitor"},
    "msg_monitor_script_not_found": {
        "en": "Monitor script not found:\n{path}",
        "pt": "Script do monitor não encontrado:\n{path}",
    },
    "msg_monitor_open_error": {
        "en": "Could not open the monitor window:\n{error}",
        "pt": "Não foi possível abrir a janela do monitor:\n{error}",
    },

    # ---------------------------------------------------------------- HOME tab
    "home_tagline": {
        "en": "Deep learning applied to the Ligands Generation Pipeline",
        "pt": "Aprendizado profundo aplicado ao Pipeline de Geração de Ligantes",
    },
    "home_grp_cpu": {"en": "Hardware Specs — CPU", "pt": "Especificações de Hardware — CPU"},
    "home_grp_gpu": {"en": "Hardware Specs — GPU", "pt": "Especificações de Hardware — GPU"},
    "home_grp_sw": {"en": "Software Specs", "pt": "Especificações de Software"},
    "home_grp_pipeline": {"en": "Pipeline — Steps", "pt": "Pipeline — Etapas"},
    "home_step1_name": {"en": "Step 1 – Job", "pt": "Etapa 1 – Job"},
    "home_step1_desc": {
        "en": "Working directory setup, job naming and path validation",
        "pt": "Configuração do diretório de trabalho, nomeação do job e validação de caminhos",
    },
    "home_step2_name": {"en": "Step 2 – Collection", "pt": "Etapa 2 – Coleta"},
    "home_step2_desc": {
        "en": "PDB download (public datasets + user-defined targets)",
        "pt": "Download de PDB (datasets públicos + alvos definidos pelo usuário)",
    },
    "home_step3_name": {"en": "Step 3 – Curation", "pt": "Etapa 3 – Curadoria"},
    "home_step3_desc": {
        "en": "Structure validation, binding site extraction",
        "pt": "Validação de estrutura, extração do sítio de ligação",
    },
    "home_step4_name": {"en": "Step 4 – Exploration", "pt": "Etapa 4 – Exploração"},
    "home_step4_desc": {
        "en": "Dataset statistics and visualization",
        "pt": "Estatísticas e visualização do dataset",
    },
    "home_step5_name": {"en": "Step 5 – Voxelization", "pt": "Etapa 5 – Voxelização"},
    "home_step5_desc": {
        "en": "3D tensors + molecular graphs (PyG)",
        "pt": "Tensores 3D + grafos moleculares (PyG)",
    },
    "home_step6_name": {"en": "Step 6 – Training", "pt": "Etapa 6 – Treinamento"},
    "home_step6_desc": {
        "en": "CVAE: 3D-CNN encoder + GNN decoder",
        "pt": "CVAE: encoder 3D-CNN + decoder GNN",
    },
    "home_step7_name": {"en": "Step 7 – Evaluation", "pt": "Etapa 7 – Avaliação"},
    "home_step7_desc": {
        "en": "Diagnostics, metrics and training curves",
        "pt": "Diagnósticos, métricas e curvas de treinamento",
    },
    "home_step8_name": {"en": "Step 8 – Generation", "pt": "Etapa 8 – Geração"},
    "home_step8_desc": {
        "en": "Ligand candidates for a target protein",
        "pt": "Candidatos a ligante para uma proteína-alvo",
    },
    "home_btn_start": {"en": "Start", "pt": "Iniciar"},
    "home_btn_install": {"en": "Install Requirements", "pt": "Instalar Dependências"},

    # ---------------------------------------------------------------- Generic / reused across tabs
    "lbl_phase": {"en": "Phase:", "pt": "Fase:"},
    "lbl_workers_auto": {"en": "Workers (0=auto):", "pt": "Trabalhadores (0=auto):"},
    "btn_browse": {"en": "Browse...", "pt": "Procurar..."},
    "msg_warning_title": {"en": "Warning", "pt": "Atenção"},
    "msg_error_title": {"en": "Error", "pt": "Erro"},
    "dlg_checkpoints_folder": {"en": "Checkpoints folder", "pt": "Pasta de checkpoints"},
    "ph_checkpoints_default": {"en": "checkpoints/ (default)", "pt": "checkpoints/ (padrão)"},
    "nav_back": {"en": " << BACK ", "pt": " << VOLTAR "},
    "nav_next": {"en": " NEXT >> ", "pt": " AVANÇAR >> "},

    # ---------------------------------------------------------------- Generic pipeline tab (_build_pipeline_tab)
    "sec_execution_log": {"en": "Execution Log", "pt": "Log de Execução"},
    "btn_run": {"en": "▶  Run", "pt": "▶  Executar"},
    "btn_stop": {"en": "⏹  Stop", "pt": "⏹  Parar"},
    "msg_please_start_job": {
        "en": "Please configure and start a Job (Step 1) first.",
        "pt": "Configure e inicie um Job primeiro (Etapa 1).",
    },
    "msg_config_error_title": {"en": "Configuration Error", "pt": "Erro de Configuração"},

    # ---------------------------------------------------------------- STEP 1 — Job configuration
    "title_step1": {"en": "Step 1 — Job Configuration", "pt": "Etapa 1 — Configuração do Job"},
    "s1_grp_root": {"en": "CODEEP Root Directory", "pt": "Diretório Raiz do CODEEP"},
    "s1_grp_job_id": {"en": "Job Identification", "pt": "Identificação do Job"},
    "s1_ph_job_name": {"en": "e.g.: KinaseInhibitors_run01", "pt": "ex.: KinaseInhibitors_run01"},
    "s1_tooltip_date": {
        "en": "Date used to compose the Job ID.\nChange this to resume a job started on a "
              "previous day\nwithout renaming the folder manually.",
        "pt": "Data usada para compor o ID do Job.\nAltere isso para retomar um job iniciado em "
              "um dia anterior\nsem precisar renomear a pasta manualmente.",
    },
    "s1_lbl_job_name": {"en": "Job Name:", "pt": "Nome do Job:"},
    "s1_lbl_date": {"en": "Date:", "pt": "Data:"},
    "s1_lbl_job_id": {"en": "Job ID:", "pt": "ID do Job:"},
    "s1_grp_search": {
        "en": "Phase 2 / 3 — Target Search Terms  (one per line)",
        "pt": "Fase 2 / 3 — Termos de Busca do Alvo  (um por linha)",
    },
    "s1_sec_org_terms": {
        "en": "Organism terms  (Phase 2 — RCSB taxonomy search)",
        "pt": "Termos de organismo  (Fase 2 — busca por taxonomia no RCSB)",
    },
    "s1_ph_org_terms": {
        "en": "e.g.:\nHomo sapiens\nRattus norvegicus\nMus musculus",
        "pt": "ex.:\nHomo sapiens\nRattus norvegicus\nMus musculus",
    },
    "s1_sec_desc_terms": {
        "en": "Entity description terms  (Phase 2 — RCSB keyword search)",
        "pt": "Termos de descrição da entidade  (Fase 2 — busca por palavra-chave no RCSB)",
    },
    "s1_ph_desc_terms": {
        "en": "e.g.:\nprotein kinase\ntyrosine kinase\nATP-binding",
        "pt": "ex.:\nprotein kinase\ntyrosine kinase\nATP-binding",
    },
    "s1_sec_p3_ids": {
        "en": "Phase 3 — Specific PDB IDs  (comma or newline separated)",
        "pt": "Fase 3 — IDs de PDB específicos  (separados por vírgula ou quebra de linha)",
    },
    "s1_ph_p3_ids": {
        "en": "e.g.:\n1ATP, 2SRC, 3EKL\n4HJB\n5J5X",
        "pt": "ex.:\n1ATP, 2SRC, 3EKL\n4HJB\n5J5X",
    },
    "s1_grp_paths": {"en": "Detected Paths", "pt": "Caminhos Detectados"},
    "s1_sep_global": {"en": "Global (shared — Phase 1)", "pt": "Global (compartilhado — Fase 1)"},
    "s1_sep_job": {"en": "Job-specific (Phase 2 / 3)", "pt": "Específico do Job (Fase 2 / 3)"},
    "s1_placeholder_no_job": {
        "en": "— (set job name first) —", "pt": "— (defina o nome do job primeiro) —",
    },
    "s1_btn_create_job": {"en": "Create / Load Job", "pt": "Criar / Carregar Job"},
    "s1_msg_job_ready": {"en": "✅  Job ready:  {job_dir}", "pt": "✅  Job pronto:  {job_dir}"},
    "s1_msg_fill_root_name": {
        "en": "Please fill in the CODEEP root directory and a job name.",
        "pt": "Preencha o diretório raiz do CODEEP e um nome de job.",
    },
    "dlg_select_root_title": {
        "en": "Select the CODEEP root folder", "pt": "Selecione a pasta raiz do CODEEP",
    },

    # ---------------------------------------------------------------- STEP 2 — Collection
    "title_step2": {
        "en": "Step 2 — Data Collection (PDB / Datasets)",
        "pt": "Etapa 2 — Coleta de Dados (PDB / Datasets)",
    },
    "s2_lbl_dataset": {"en": "Phase 1 Dataset:", "pt": "Dataset da Fase 1:"},
    "s2_lbl_phase": {"en": "Collection Phase:", "pt": "Fase de Coleta:"},
    "s2_phase1": {"en": "Phase 1", "pt": "Fase 1"},
    "s2_phase2": {"en": "Phase 2", "pt": "Fase 2"},
    "s2_phase3": {"en": "Phase 3", "pt": "Fase 3"},
    "s2_phase_all": {"en": "All Phases", "pt": "Todas as Fases"},
    "s2_tooltip_phase": {
        "en": "Phase 1 — Download the external dataset. (lp-PDBbind / RCSB / PLINDER / MOAD)\n"
              "Phase 2 — Search RCSB by organism/description (terms defined in Step 1)\n"
              "Phase 3 — Download specific PDBs by ID (IDs defined in Step 1)\n"
              "All Phases — Execute all three phases in sequence",
        "pt": "Fase 1 — Baixa o dataset externo. (lp-PDBbind / RCSB / PLINDER / MOAD)\n"
              "Fase 2 — Busca no RCSB por organismo/descrição (termos definidos na Etapa 1)\n"
              "Fase 3 — Baixa PDBs específicos por ID (IDs definidos na Etapa 1)\n"
              "Todas as Fases — Executa as três fases em sequência",
    },
    "s2_lbl_max_entries": {"en": "Max Phase 1 entries:", "pt": "Máx. de entradas da Fase 1:"},
    "s2_lbl_moad_dir": {"en": "MOAD Dir:", "pt": "Pasta MOAD:"},
    "s2_ph_moad_dir": {
        "en": "Required only for MOAD dataset", "pt": "Obrigatório apenas para o dataset MOAD",
    },
    "s2_chk_status": {
        "en": "Only show collected data status", "pt": "Mostrar apenas o status dos dados coletados",
    },
    "s2_dlg_phase2_title": {
        "en": "Phase 2 — Search terms not configured",
        "pt": "Fase 2 — Termos de busca não configurados",
    },
    "s2_dlg_phase2_body": {
        "en": "Phase 2 requires organism or description search terms (configured in Step 1), "
              "but both fields are empty.\n\nWithout search terms, Phase 2 cannot retrieve "
              "target-specific structures from RCSB.\n\nWould you like to skip Phase 2 and "
              "continue with the other selected phases?",
        "pt": "A Fase 2 requer termos de busca por organismo ou descrição (configurados na "
              "Etapa 1), mas os dois campos estão vazios.\n\nSem termos de busca, a Fase 2 não "
              "consegue obter estruturas específicas do alvo no RCSB.\n\nDeseja pular a Fase 2 "
              "e continuar com as outras fases selecionadas?",
    },
    "s2_err_phase2_cancelled": {
        "en": "Cancelled. Fill in organism or description terms in Step 1 and try again.",
        "pt": "Cancelado. Preencha os termos de organismo ou descrição na Etapa 1 e tente "
              "novamente.",
    },
    "s2_dlg_phase3_title": {
        "en": "Phase 3 — PDB IDs not configured", "pt": "Fase 3 — IDs de PDB não configurados",
    },
    "s2_dlg_phase3_body": {
        "en": "Phase 3 requires a list of specific PDB IDs (configured in Step 1), but the "
              "field is empty.\n\nWithout PDB IDs, Phase 3 cannot download the target "
              "structures.\n\nWould you like to skip Phase 3 and continue with the other "
              "selected phases?",
        "pt": "A Fase 3 requer uma lista de IDs de PDB específicos (configurados na Etapa 1), "
              "mas o campo está vazio.\n\nSem IDs de PDB, a Fase 3 não consegue baixar as "
              "estruturas do alvo.\n\nDeseja pular a Fase 3 e continuar com as outras fases "
              "selecionadas?",
    },
    "s2_err_phase3_cancelled": {
        "en": "Cancelled. Fill in the PDB IDs in Step 1 and try again.",
        "pt": "Cancelado. Preencha os IDs de PDB na Etapa 1 e tente novamente.",
    },
    "s2_err_no_phase_active": {
        "en": "No collection phase active after validation. Please check your configuration "
              "in Step 1 and try again.",
        "pt": "Nenhuma fase de coleta ativa após a validação. Verifique sua configuração na "
              "Etapa 1 e tente novamente.",
    },
    "dlg_moad_folder": {"en": "MOAD folder", "pt": "Pasta do MOAD"},

    # ---------------------------------------------------------------- STEP 3 — Curation
    "title_step3": {
        "en": "Step 3 — Structure Validation and Curation",
        "pt": "Etapa 3 — Validação de Estrutura e Curadoria",
    },
    "s3_tooltip_workers": {"en": "0 = detect automatically", "pt": "0 = detectar automaticamente"},
    "s3_lbl_cutoff": {"en": "Site cutoff (Å):", "pt": "Cutoff do sítio (Å):"},
    "s3_lbl_min_site": {"en": "Min. site residues:", "pt": "Resíduos mín. do sítio:"},
    "s3_lbl_max_mw": {"en": "Max. ligand MW (Da):", "pt": "MW máx. do ligante (Da):"},
    "s3_lbl_min_heavy": {"en": "Min. heavy atoms:", "pt": "Átomos pesados mín.:"},

    # ---------------------------------------------------------------- STEP 4 — Exploration
    "title_step4": {
        "en": "Step 4 — Database Exploration", "pt": "Etapa 4 — Exploração do Banco de Dados",
    },
    "s4_lbl_desc": {
        "en": "Generates statistics and the exploration dashboard from the curated dataset.\n"
              "Saves summary_dashboard.png and .svg to data/figures/.",
        "pt": "Gera estatísticas e o painel de exploração a partir do dataset curado.\nSalva "
              "summary_dashboard.png e .svg em data/figures/.",
    },
    "s4_btn_view_dash": {"en": "View Dashboard", "pt": "Ver Painel"},

    # ---------------------------------------------------------------- STEP 5 — Voxelization
    "title_step5": {
        "en": "Step 5 — 3D Voxelization + Molecular Graphs",
        "pt": "Etapa 5 — Voxelização 3D + Grafos Moleculares",
    },
    "s5_ph_pair": {"en": "empty = process all pairs", "pt": "vazio = processa todos os pares"},
    "s5_tooltip_vox_workers": {
        "en": "0 = automatic (recommended = 1 for CUDA)",
        "pt": "0 = automático (recomendado = 1 para CUDA)",
    },
    "s5_tooltip_max_qm": {
        "en": "Heavy-atom limit for full QM (RHF/STO-3G).\nUsed both for the ligand (not wired "
              "into the tensor yet) and\nfor EACH SITE RESIDUE in the 'RHF/STO-3G per residue' "
              "method\n(a protonated residue rarely exceeds ~25 atoms, so the\ndefault of 60 is "
              "almost always enough).",
        "pt": "Limite de átomos pesados para QM completo (RHF/STO-3G).\nUsado tanto para o "
              "ligante (não conectado ao tensor hoje) quanto\npara CADA RESÍDUO do sítio no "
              "método 'RHF/STO-3G per residue'\n(um resíduo protonado raramente passa de ~25 "
              "átomos, então o\npadrão de 60 quase sempre é suficiente).",
    },
    "s5_tooltip_vox_batch": {
        "en": "Atoms per batch in Gaussian voxelization.\nCuPy (GPU): 500 recommended "
              "(RTX 8–16 GB).\nNumPy (CPU): 100 recommended.\nHigher values reduce kernel "
              "overhead but increase VRAM usage.\nDetected preset: {batch} ({cupy_status})",
        "pt": "Átomos por lote na voxelização gaussiana.\nCuPy (GPU): 500 recomendado "
              "(RTX 8–16 GB).\nNumPy (CPU): 100 recomendado.\nValores maiores reduzem overhead "
              "de kernel mas aumentam uso de VRAM.\nPreset detectado: {batch} ({cupy_status})",
    },
    "s5_cupy_available": {"en": "CuPy available", "pt": "CuPy disponível"},
    "s5_cupy_unavailable": {"en": "CuPy unavailable", "pt": "CuPy indisponível"},
    "s5_chk_ecfp4": {
        "en": "Use ECFP4 in graph (reduces diversity)",
        "pt": "Usar ECFP4 no grafo (reduz diversidade)",
    },
    "s5_chk_protonate_site": {
        "en": "Protonate site residues (pH-based)", "pt": "Protonar resíduos do sítio (baseado em pH)",
    },
    "s5_tooltip_protonate_site": {
        "en": "Adds explicit H atoms to site residues based on standard pKa\nand the pH below "
              "(Asp/Glu/His/Cys/Tyr/Lys/Arg). Also replaces the\nper-element approximate charge "
              "with real per-atom Gasteiger charges.\nTurning this off reverts to the legacy "
              "behavior (no H, approximate charge).",
        "pt": "Adiciona H explícitos aos resíduos do sítio conforme pKa padrão\ne o pH abaixo "
              "(Asp/Glu/His/Cys/Tyr/Lys/Arg). Também troca a carga\naproximada por elemento por "
              "cargas Gasteiger reais por átomo.\nDesligar volta ao comportamento legado (sem H, "
              "carga aproximada).",
    },
    "s5_tooltip_site_ph": {
        "en": "pH used to decide the protonation state of the site's\nionizable residues "
              "(threshold rule over table pKa values).",
        "pt": "pH usado para decidir o estado de protonação dos resíduos\nionizáveis do sítio "
              "(regra de limiar sobre pKa de tabela).",
    },
    "qm_method_rhf_fragments": {
        "en": "RHF/STO-3G per residue (fragmented)",
        "pt": "RHF/STO-3G por resíduo (fragmentado)",
    },
    "qm_method_gfn_xtb": {
        "en": "GFN-xTB (whole site, requires xtb binary)",
        "pt": "GFN-xTB (sítio inteiro, requer o binário xtb)",
    },
    "qm_method_heuristic": {
        "en": "Heuristic (electronegativity, no QM)", "pt": "Heurística (eletronegatividade, sem QM)",
    },
    "s5_tooltip_site_qm_method": {
        "en": "Level of theory for channels 3-4 (site donor/acceptor field):\n  RHF/STO-3G per "
              "residue — real QM, FMO-style fragmented (slower,\n    ~1s/residue; uses the "
              "already-installed PySCF/GPU4PySCF).\n  GFN-xTB on the whole site — "
              "semi-empirical, faster, requires the\n    `xtb` binary on PATH (not installed by "
              "default).\n  Heuristic — electronegativity + charge, no QM (fastest,\n    also "
              "the automatic fallback when the other two fail).",
        "pt": "Nível de teoria para os canais 3-4 (campo doador/aceptor do sítio):\n  "
              "RHF/STO-3G por resíduo — QM real, fragmentado tipo FMO (mais lento,\n    "
              "~1s/resíduo; usa PySCF/GPU4PySCF já instalado).\n  GFN-xTB no sítio inteiro — "
              "semi-empírico, mais rápido, requer o\n    binário `xtb` no PATH (não instalado "
              "por padrão).\n  Heurística — eletronegatividade + carga, sem QM (mais rápido,\n"
              "    também o fallback automático quando os outros dois falham).",
    },
    "s5_lbl_pair": {"en": "Specific pair:", "pt": "Par específico:"},
    "s5_lbl_grid": {"en": "Grid size (N³):", "pt": "Tamanho da grade (N³):"},
    "s5_lbl_padding": {"en": "Padding (Å):", "pt": "Preenchimento (Å):"},
    "s5_lbl_max_qm": {"en": "Max QM atoms:", "pt": "Máx. átomos QM:"},
    "s5_lbl_batch": {"en": "Vox batch size:", "pt": "Tamanho do lote de voxel.:"},
    "s5_lbl_qm_basis": {"en": "QM basis:", "pt": "Base QM:"},
    "s5_tooltip_qm_basis": {
        "en": "Only affects the 'RHF/STO-3G per residue' method above — ignored by GFN-xTB "
              "(its own fixed semi-empirical parameterization, no basis set) and by Heuristic "
              "(no QM at all). Changing this away from 'sto-3g' means the actual calculation is "
              "no longer STO-3G despite the method's name (e.g. picking '3-21g' here runs real "
              "RHF/3-21G) — larger bases are more accurate but slower per residue.",
        "pt": "Só afeta o método 'RHF/STO-3G por resíduo' acima — ignorado pelo GFN-xTB "
              "(parametrização semi-empírica própria, fixa, sem base gaussiana) e pela "
              "Heurística (sem QM nenhum). Mudar isso para além de 'sto-3g' faz o cálculo real "
              "deixar de ser STO-3G apesar do nome do método (ex.: escolher '3-21g' aqui roda "
              "RHF/3-21G de verdade) — bases maiores são mais precisas, porém mais lentas por "
              "resíduo.",
    },
    "s5_lbl_site_ph": {"en": "Site pH:", "pt": "pH do sítio:"},
    "s5_lbl_site_qm_method": {
        "en": "Site channels 3-4 method:", "pt": "Método dos canais 3-4 do sítio:",
    },

    # ---------------------------------------------------------------- STEP 6 — Training
    "title_step6": {"en": "Step 6 — CVAE / GNN Training", "pt": "Etapa 6 — Treinamento CVAE / GNN"},
    "s6_lbl_epochs": {"en": "Epochs:", "pt": "Épocas:"},
    "s6_lbl_batch": {"en": "Batch size:", "pt": "Tamanho do lote:"},
    "s6_lbl_latent": {"en": "Latent dim:", "pt": "Dim. latente:"},
    "s6_lbl_hidden": {"en": "Hidden dim:", "pt": "Dim. oculta:"},
    "s6_lbl_gnn_layers": {"en": "GNN layers:", "pt": "Camadas GNN:"},
    "s6_lbl_lr": {"en": "Learning rate:", "pt": "Taxa de aprendizado:"},
    "s6_lbl_beta_max": {"en": "Beta KL (max):", "pt": "Beta KL (máx.):"},
    "s6_lbl_beta_warmup": {"en": "Beta warmup:", "pt": "Beta warmup:"},
    "s6_lbl_dropout": {"en": "Dropout:", "pt": "Dropout:"},
    "s6_lbl_patience": {"en": "Patience:", "pt": "Paciência:"},
    "s6_lbl_device": {"en": "Device:", "pt": "Dispositivo:"},
    "s6_lbl_resume": {"en": "Resume from:", "pt": "Retomar de:"},
    "s6_lbl_ckpt_out": {"en": "Save ckpt to:", "pt": "Salvar checkpoint em:"},
    "s6_ph_resume": {
        "en": "optional — best_model.pt from previous phase",
        "pt": "opcional — best_model.pt da fase anterior",
    },
    "s6_chk_freeze_enc": {
        "en": "Freeze encoder (Phase 3 fine-tuning)",
        "pt": "Congelar encoder (fine-tuning da Fase 3)",
    },
    "s6_dlg_select_best_model": {"en": "Select best_model.pt", "pt": "Selecione best_model.pt"},
    "s6_err_tensor_not_found": {
        "en": "Tensor dataset not found for Phase {phase}: {tensor_dir}",
        "pt": "Dataset de tensores não encontrado para a Fase {phase}: {tensor_dir}",
    },

    # ---------------------------------------------------------------- STEP 7 — Evaluation
    "title_step7": {
        "en": "Step 7 — Trained Model Evaluation", "pt": "Etapa 7 — Avaliação do Modelo Treinado",
    },
    "s7_chk_do_generate": {
        "en": "Generate ligands during evaluation", "pt": "Gerar ligantes durante a avaliação",
    },
    "s7_lbl_checkpoints": {"en": "Checkpoints:", "pt": "Checkpoints:"},
    "s7_lbl_figures": {"en": "Figures:", "pt": "Figuras:"},
    "s7_btn_training_curves": {"en": "Training Curves", "pt": "Curvas de Treinamento"},
    "s7_btn_performance_report": {"en": "Performance Report", "pt": "Relatório de Desempenho"},

    # ---------------------------------------------------------------- STEP 8 — Generation
    "title_step8": {
        "en": "Step 8 — Ligand Candidate Generation", "pt": "Etapa 8 — Geração de Candidatos a Ligante",
    },
    "s8_ph_pdb": {
        "en": "Target protein PDB file (required)", "pt": "Arquivo PDB da proteína-alvo (obrigatório)",
    },
    "s8_ph_site_pdb": {
        "en": "optional — PDB with only binding-site residues",
        "pt": "opcional — PDB apenas com os resíduos do sítio de ligação",
    },
    "s8_ph_out": {
        "en": "JOBS/<job_id>/results/{pdb}/ (default)",
        "pt": "JOBS/<job_id>/results/{pdb}/ (padrão)",
    },
    "s8_ph_ligand_ref": {
        "en": "optional — reference ligand SDF", "pt": "opcional — SDF do ligante de referência",
    },
    "s8_lbl_model_phase": {"en": "Model phase:", "pt": "Fase do modelo:"},
    "s8_lbl_protein_pdb": {"en": "Protein (PDB):", "pt": "Proteína (PDB):"},
    "s8_lbl_site_pdb": {"en": "Site residues (PDB):", "pt": "Resíduos do sítio (PDB):"},
    "s8_lbl_output": {"en": "Output:", "pt": "Saída:"},
    "s8_lbl_ref_ligand": {"en": "Ref. ligand:", "pt": "Ligante de ref.:"},
    "s8_lbl_n_samples": {"en": "No. samples:", "pt": "Nº de amostras:"},
    "s8_lbl_temperature": {"en": "Temperature:", "pt": "Temperatura:"},
    "s8_lbl_cutoff": {"en": "Site cutoff (Å):", "pt": "Cutoff do sítio (Å):"},
    "s8_lbl_top_n": {"en": "Top N report:", "pt": "Relatório Top N:"},
    "s8_lbl_min_atoms": {"en": "Min. site atoms:", "pt": "Átomos mín. do sítio:"},
    "s8_chk_protonate_site": {
        "en": "Protonate site residues (pH-based)", "pt": "Protonar resíduos do sítio (baseado em pH)",
    },
    "s8_tooltip_gen_qm_method": {
        "en": "Fallback only: when the selected checkpoint has its training voxelization "
              "config saved (checkpoints trained after this feature was added), that value is "
              "always used instead, with a log warning if it differs from this field — this "
              "choice only takes effect for checkpoints trained before that (no config saved).",
        "pt": "Só como reserva: quando o checkpoint escolhido tem a config de voxelização do "
              "treino salva (checkpoints treinados depois desta funcionalidade existir), esse "
              "valor é sempre usado no lugar deste, com aviso no log se divergir — esta escolha "
              "só tem efeito para checkpoints treinados antes disso (sem config salva).",
    },
    "s8_tooltip_gen_qm_basis": {
        "en": "Fallback only (same rule as the site channels 3-4 method above): only takes "
              "effect for checkpoints trained before the voxelization config started being "
              "saved automatically — only matters for the RHF/STO-3G per residue method.",
        "pt": "Só como reserva (mesma regra do método dos canais 3-4 acima): só tem efeito "
              "para checkpoints treinados antes de a config de voxelização passar a ser salva "
              "automaticamente — só importa para o método RHF/STO-3G por resíduo.",
    },
    "ff_mmff_uff": {
        "en": "MMFF/UFF (RDKit, no extra dependency)", "pt": "MMFF/UFF (RDKit, sem dependência extra)",
    },
    "ff_amber_ff14sb": {
        "en": "AMBER ff14SB (protein FF, requires GROMACS+AmberTools)",
        "pt": "AMBER ff14SB (FF de proteína, requer GROMACS+AmberTools)",
    },
    "ff_charmm36": {
        "en": "CHARMM36 (protein FF, requires GROMACS+AmberTools)",
        "pt": "CHARMM36 (FF de proteína, requer GROMACS+AmberTools)",
    },
    "s8_lbl_final_ff": {"en": "Final geometry FF:", "pt": "FF de geometria final:"},
    "s8_tooltip_final_ff": {
        "en": "Force field for the final minimization of the COMPLEX (ligand +\nsite, not the "
              "isolated ligand) before the steric clash check\nagainst the protein.\n\nMMFF/UFF "
              "(RDKit) — default, no extra dependency. UFF covers\nsite metals (Zn/Fe/Mg) that "
              "MMFF94 does not support well.\n\nAMBER ff14SB / CHARMM36 — real protein force "
              "field via\nGROMACS (gmx pdb2gmx), ligand parametrized as GAFF2 via\n"
              "acpype/AmberTools (antechamber+parmchk2+tleap). More rigorous,\nrequires the "
              "`gmx` binary (already used by CODYN) + `acpype` (pip) +\nAmberTools "
              "(antechamber/parmchk2/tleap — conda-forge) on PATH.\nAutomatically falls back to "
              "MMFF/UFF with a log warning if missing.\nUses the amber99sb-ildn.ff/charmm27.ff "
              "folder already bundled with\nGROMACS by default (not the newer ff14SB/CHARMM36 "
              "files,\nwhich do not ship with the standard installation).",
        "pt": "Campo de força para a minimização final do COMPLEXO (ligante +\nsítio, não o "
              "ligante isolado) antes da checagem de colisão\nestérica com a proteína.\n\n"
              "MMFF/UFF (RDKit) — padrão, sem dependência extra. UFF cobre\nmetais do sítio "
              "(Zn/Fe/Mg) que o MMFF94 não suporta bem.\n\nAMBER ff14SB / CHARMM36 — campo de "
              "força de proteína real via\nGROMACS (gmx pdb2gmx), ligante parametrizado como "
              "GAFF2 via\nacpype/AmberTools (antechamber+parmchk2+tleap). Mais rigoroso,\n"
              "requer o binário `gmx` (já usado pelo CODYN) + `acpype` (pip) +\nAmberTools "
              "(antechamber/parmchk2/tleap — conda-forge) no PATH.\nCai automaticamente para "
              "MMFF/UFF com aviso no log se ausentes.\nUsa a pasta "
              "amber99sb-ildn.ff/charmm27.ff já empacotada com o\nGROMACS por padrão (não os "
              "arquivos ff14SB/CHARMM36 mais\nrecentes, que não vêm com a instalação padrão).",
    },
    "s8_chk_induced_fit": {
        "en": "Induced-fit (flexible side chains)", "pt": "Induced-fit (cadeias laterais flexíveis)",
    },
    "s8_tooltip_induced_fit": {
        "en": "Unchecked (default): site 100% rigid during the final minimization —\nonly the "
              "ligand moves.\n\nChecked: locks only the backbone of the site residues "
              "(N, Cα, C, O —\nrepresents where the rest of the real protein would hold this\n"
              "residue in place, since we only have the cropped pocket, not the\nwhole chain). "
              "The side chain is free to adjust to the ligand during\nthe same minimization "
              "(not a separate thermal dynamics run). Works\nwith both force fields (MMFF/UFF "
              "and AMBER/CHARMM).",
        "pt": "Desmarcado (padrão): sítio 100% rígido na minimização final —\nsó o ligante se "
              "move.\n\nMarcado: trava só o backbone dos resíduos do sítio (N, Cα, C, O —\n"
              "representa onde o resto da proteína real prenderia esse resíduo,\njá que só "
              "temos o bolso recortado, não a cadeia inteira). A cadeia\nlateral fica livre "
              "para se ajustar ao ligante durante a mesma\nminimização (não é uma dinâmica "
              "térmica separada). Funciona nos\ndois campos de força (MMFF/UFF e AMBER/CHARMM).",
    },
    "s8_err_invalid_pdb": {
        "en": "Please select a valid PDB file for the target protein.",
        "pt": "Selecione um arquivo PDB válido para a proteína-alvo.",
    },
    "s8_err_invalid_site_pdb": {
        "en": "Please select a valid PDB file containing only the binding-site residues.",
        "pt": "Selecione um arquivo PDB válido contendo apenas os resíduos do sítio de ligação.",
    },
    "s8_err_no_checkpoint": {
        "en": "No trained checkpoint found for Phase {phase}.",
        "pt": "Nenhum checkpoint treinado encontrado para a Fase {phase}.",
    },
    "s8_dlg_select_pdb": {"en": "Select PDB file", "pt": "Selecione o arquivo PDB"},
    "s8_dlg_select_site_pdb": {
        "en": "Select binding-site residues PDB",
        "pt": "Selecione o PDB dos resíduos do sítio de ligação",
    },
    "s8_dlg_output_folder": {"en": "Output folder", "pt": "Pasta de saída"},
    "s8_dlg_reference_ligand": {"en": "Reference ligand", "pt": "Ligante de referência"},

    # ---------------------------------------------------------------- View-figure helpers / About
    "msg_figure_not_found": {"en": "Figure not found", "pt": "Figura não encontrada"},
    "msg_run_step_first": {
        "en": "Run the corresponding step first.\n\nExpected at:\n{fig_dir}",
        "pt": "Execute a etapa correspondente primeiro.\n\nEsperado em:\n{fig_dir}",
    },
    "msg_could_not_load": {
        "en": "Could not load:\n{path}", "pt": "Não foi possível carregar:\n{path}",
    },
    "btn_open_in_system": {"en": "Open in system", "pt": "Abrir no sistema"},
    "btn_close": {"en": "Close", "pt": "Fechar"},
    "msg_summary_dashboard_title": {
        "en": "Summary Dashboard — Phase {phase}", "pt": "Painel Resumo — Fase {phase}",
    },
    "msg_run_step4_first": {
        "en": "Run Step 4 with Phase {phase} first.\n\nExpected at:\n{fig_dir}",
        "pt": "Execute a Etapa 4 com a Fase {phase} primeiro.\n\nEsperado em:\n{fig_dir}",
    },
    "msg_fig_title_with_phase": {
        "en": "{title} — Phase {phase}", "pt": "{title} — Fase {phase}",
    },
    "msg_run_step7_first": {
        "en": "Run Step 7 with Phase {phase} first.\n\nExpected at:\n{fig_dir}",
        "pt": "Execute a Etapa 7 com a Fase {phase} primeiro.\n\nEsperado em:\n{fig_dir}",
    },
    "msg_module_not_found_title": {"en": "Module not found", "pt": "Módulo não encontrado"},
    "msg_module_not_found_body": {
        "en": "CODEEP_requirements.py was not found in the CODEEP folder.",
        "pt": "CODEEP_requirements.py não foi encontrado na pasta do CODEEP.",
    },
    "about_title": {"en": "About CODEEP", "pt": "Sobre o CODEEP"},
    "progress_label": {"en": "Progress", "pt": "Progresso"},
    "progress_eta_label": {"en": "ETA", "pt": "Restante"},
    "progress_done": {"en": "Done", "pt": "Concluído"},
    "log_completed": {"en": "\n✅ Completed!", "pt": "\n✅ Concluído!"},
    "log_interrupted": {
        "en": "\n❌ Interrupted or finished with errors.",
        "pt": "\n❌ Interrompido ou finalizado com erros.",
    },
    "about_body": {
        "en": "CODEEP\nDeep Learning Ligand Generation Pipeline\n\nPart of the suite:\n   "
              "CODOC · CODYN · CODEEP · CODRUG\n\nDeveloped by:\n   Moisés Maia Neto\n   "
              "CEB / Universidade Federal do Paraná - UFPR\n\n   Brasil\n\n"
              "Contact: moimaian@gmail.com\n\n",
        "pt": "CODEEP\nPipeline de Geração de Ligantes por Aprendizado Profundo\n\nParte do "
              "conjunto:\n   CODOC · CODYN · CODEEP · CODRUG\n\nDesenvolvido por:\n   Moisés "
              "Maia Neto\n   CEB / Universidade Federal do Paraná - UFPR\n\n   Brasil\n\n"
              "Contato: moimaian@gmail.com\n\n",
    },
}


def t(chave, idioma, **kwargs):
    valor = _TEXTOS.get(chave, {}).get(idioma)
    if valor is None:
        valor = _TEXTOS.get(chave, {}).get(IDIOMA_PADRAO, chave)
    return valor.format(**kwargs) if kwargs else valor
