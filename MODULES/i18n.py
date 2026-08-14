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
    "menu_step2": {"en": "Step 2 – Data Management", "pt": "Etapa 2 – Gerenciamento de Dados"},
    "menu_step3": {"en": "Step 3 – Voxelization", "pt": "Etapa 3 – Voxelização"},
    "menu_step4": {"en": "Step 4 – Hyperparameters", "pt": "Etapa 4 – Hiperparâmetros"},
    "menu_step5": {"en": "Step 5 – Training", "pt": "Etapa 5 – Treinamento"},
    "menu_step6": {"en": "Step 6 – Evaluation", "pt": "Etapa 6 – Avaliação"},
    "menu_step7": {"en": "Step 7 – Generation", "pt": "Etapa 7 – Geração"},
    "menu_exit": {"en": "Exit", "pt": "Sair"},
    "menu_help": {"en": "Help", "pt": "Ajuda"},
    "menu_install_requirements": {"en": "Install Requirements", "pt": "Instalar Dependências"},
    "menu_final_report": {"en": "Generate Final Report", "pt": "Gerar Relatório Final"},
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

    # ---------------------------------------------------------------- Falha ao construir uma aba
    "tab_load_error_title": {
        "en": "Couldn't load this tab ({aba})",
        "pt": "Não foi possível carregar esta aba ({aba})",
    },
    "tab_load_error_detail": {
        "en": "A dependency is likely missing. Open Menu → Install "
              "Requirements, install what's needed, and restart CODEEP.\n\n"
              "Technical detail: {erro}",
        "pt": "Provavelmente falta alguma dependência. Abra Menu → "
              "Instalar Dependências, instale o que faltar e reinicie o "
              "CODEEP.\n\nDetalhe técnico: {erro}",
    },

    # ---------------------------------------------------------------- HW monitor (top-right corner)
    # Monitor num PROCESSO separado (ver MODULES/hw_monitor_window.py), mesmo padrão do CODRUG —
    # só um botão aqui na janela principal; os textos ao vivo (CPU/GPU/temperatura) ficam dentro
    # do próprio hw_monitor_window.py, fora do escopo deste toggle (janela/processo distinto).
    "btn_cpu_gpu_monitor": {"en": "CPU/GPU Monitor", "pt": "Monitor de CPU/GPU"},
    "tooltip_cpu_gpu_monitor": {
        "en": "Open a separate CPU/GPU monitor window that keeps updating even while the main "
              "window is busy running a heavy task (e.g. Step 5 Training or Step 7 Generation).",
        "pt": "Abre uma janela separada de monitor de CPU/GPU que continua atualizando mesmo "
              "quando a janela principal estiver ocupada rodando uma tarefa pesada (ex.: Etapa 5 "
              "Treinamento ou Etapa 7 Geração).",
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
    "home_step2_name": {"en": "Step 2 – Data Management", "pt": "Etapa 2 – Gerenciamento de Dados"},
    "home_step2_desc": {
        "en": "Collection, structure validation/curation and dataset exploration",
        "pt": "Coleta, validação/curadoria de estrutura e exploração do dataset",
    },
    "home_step3_name": {"en": "Step 3 – Voxelization", "pt": "Etapa 3 – Voxelização"},
    "home_step3_desc": {
        "en": "3D tensors + molecular graphs (PyG)",
        "pt": "Tensores 3D + grafos moleculares (PyG)",
    },
    "home_step4_name": {
        "en": "Step 4 – Hyperparameter optimization",
        "pt": "Etapa 4 – Otimização de hiperparâmetros",
    },
    "home_step4_desc": {
        "en": "Optuna search on a cheap data subset (optional)",
        "pt": "Busca com Optuna em um subconjunto barato dos dados (opcional)",
    },
    "home_step5_name": {"en": "Step 5 – Training", "pt": "Etapa 5 – Treinamento"},
    "home_step5_desc": {
        "en": "CVAE: 3D-CNN encoder + GNN decoder",
        "pt": "CVAE: encoder 3D-CNN + decoder GNN",
    },
    "home_step6_name": {"en": "Step 6 – Evaluation", "pt": "Etapa 6 – Avaliação"},
    "home_step6_desc": {
        "en": "Diagnostics, metrics and training curves",
        "pt": "Diagnósticos, métricas e curvas de treinamento",
    },
    "home_step7_name": {"en": "Step 7 – Generation", "pt": "Etapa 7 – Geração"},
    "home_step7_desc": {
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
        "s1_lbl_job_mode": {"en": "Project Mode:", "pt": "Modo do Projeto:"},
        "s1_rb_new_project": {"en": "New Project", "pt": "Novo Projeto"},
        "s1_rb_load_project": {"en": "Load Previous Project", "pt": "Carregar Projeto Anterior"},
        "s1_lbl_prev_job": {"en": "Previous Project:", "pt": "Projeto Anterior:"},
        "s1_tooltip_prev_job": {
          "en": "Select a previously created project to reload every saved tab state.",
          "pt": "Selecione um projeto já criado para recarregar o estado salvo de todas as abas.",
        },
    "s1_ph_job_name": {"en": "e.g.: KinaseInhibitors_run01", "pt": "ex.: KinaseInhibitors_run01"},
    "s1_tooltip_date": {
          "en": "Date and time used to compose the Job ID prefix.\nChange this before creating a "
              "new project if you need a custom timestamp.",
          "pt": "Data e hora usadas para compor o prefixo do ID do Job.\nAltere isso antes de "
              "criar um novo projeto se precisar de um timestamp personalizado.",
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
    "s1_msg_select_previous_job": {
        "en": "Select the CODEEP root directory and a previous project to load.",
        "pt": "Selecione o diretório raiz do CODEEP e um projeto anterior para carregar.",
    },
    "s1_msg_job_loaded": {
        "en": "✅  Project loaded:  {job_dir}",
        "pt": "✅  Projeto carregado:  {job_dir}",
    },
    "s1_msg_job_not_found": {
        "en": "Project folder not found:\n{job_dir}",
        "pt": "Pasta do projeto não encontrada:\n{job_dir}",
    },
    "dlg_select_root_title": {
        "en": "Select the CODEEP root folder", "pt": "Selecione a pasta raiz do CODEEP",
    },

    # ---------------------------------------------------------------- Final report
    "report_title_done": {"en": "Final Report", "pt": "Relatório Final"},
    "report_saved": {
        "en": "Final report saved to:\n{path}",
        "pt": "Relatório final salvo em:\n{path}",
    },

    # ---------------------------------------------------------------- STEP 2 — Data Management
    # (fusão das antigas Steps 2/3/4 — Collection/Curation/Exploration — em três grupos dentro
    # de uma única aba; ver _tab_step2/_grp_data_collection/_grp_data_curation/
    # _grp_data_exploration em CODEEP.py)
    "title_step2": {
        "en": "Step 2 — Data Management",
        "pt": "Etapa 2 — Gerenciamento de Dados",
    },
    "s2_grp_title": {"en": "Data Collection", "pt": "Coleta de Dados"},
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

    # ---------------------------------------------------------------- Data Management — grupo Curation
    "s2cur_grp_title": {
        "en": "Data Validation and Curation",
        "pt": "Validação e Curadoria de Dados",
    },
    "s2cur_tooltip_workers": {"en": "0 = detect automatically", "pt": "0 = detectar automaticamente"},
    "s2cur_lbl_cutoff": {"en": "Site cutoff (Å):", "pt": "Cutoff do sítio (Å):"},
    "s2cur_lbl_min_site": {"en": "Min. site residues:", "pt": "Resíduos mín. do sítio:"},
    "s2cur_lbl_max_mw": {"en": "Max. ligand MW (Da):", "pt": "MW máx. do ligante (Da):"},
    "s2cur_lbl_min_heavy": {"en": "Min. heavy atoms:", "pt": "Átomos pesados mín.:"},

    # ---------------------------------------------------------------- Data Management — grupo Exploration
    "s2exp_grp_title": {
        "en": "Data Exploration", "pt": "Exploração de Dados",
    },
    "s2exp_lbl_desc": {
        "en": "Generates statistics and the exploration dashboard from the curated dataset.\n"
              "Saves summary_dashboard.png and .svg to data/figures/.",
        "pt": "Gera estatísticas e o painel de exploração a partir do dataset curado.\nSalva "
              "summary_dashboard.png e .svg em data/figures/.",
    },
    "s2exp_btn_view_dash": {"en": "View Dashboard", "pt": "Ver Painel"},

    # ---------------------------------------------------------------- STEP 3 — Voxelization
    "title_step3": {
        "en": "Step 3 — 3D Voxelization + Molecular Graphs",
        "pt": "Etapa 3 — Voxelização 3D + Grafos Moleculares",
    },
    "s3_ph_pair": {"en": "empty = process all pairs", "pt": "vazio = processa todos os pares"},
    "s3_tooltip_vox_workers": {
        "en": "Number of pairs processed simultaneously — only takes effect if "
              "'Parallelize pairs' below is checked (0/1 = sequential either way).",
        "pt": "Número de pares processados simultaneamente — só tem efeito se "
              "'Paralelizar pares' abaixo estiver marcado (0/1 = sequencial de qualquer jeito).",
    },
    "s3_chk_parallel_pairs": {
        "en": "Parallelize pairs (experimental)",
        "pt": "Paralelizar pares (experimental)",
    },
    "s3_tooltip_parallel_pairs": {
        "en": "Off (default) = today's validated behavior: one pair at a time. On = "
              "processes several pairs at once (ThreadPoolExecutor, count set by the field "
              "above), splitting CPU threads between them so GFN-xTB fragments per residue "
              "don't oversubscribe the cores. Expected gain: modest (~15-25%, analysis-based, "
              "not validated on a large batch) — the QM step (the slowest part) already uses "
              "nearly all cores per pair on its own; parallelizing pairs only overlaps the "
              "remaining ~25% (protonation + GPU voxelization) with other pairs' QM.",
        "pt": "Desligado (padrão) = comportamento validado de hoje: um par por vez. Ligado = "
              "processa vários pares ao mesmo tempo (ThreadPoolExecutor, quantidade definida no "
              "campo acima), dividindo as threads de CPU entre eles para o GFN-xTB fragmentado "
              "por resíduo não disputar os núcleos sem controle. Ganho esperado: modesto "
              "(~15-25%, baseado em análise, não validado em lote grande) — a etapa de QM (a "
              "mais lenta) já usa quase todos os núcleos sozinha por par; paralelizar pares só "
              "sobrepõe os ~25% restantes (protonação + voxelização GPU) com o QM de outros "
              "pares.",
    },
    "s3_tooltip_max_qm": {
        "en": "Heavy-atom limit for full QM (RHF/STO-3G).\nUsed both for the ligand (not wired "
              "into the tensor yet) and\nfor EACH SITE RESIDUE in the 'RHF/STO-3G per residue' "
              "method\n(a protonated residue rarely exceeds ~25 atoms, so the\ndefault of 60 is "
              "almost always enough).",
        "pt": "Limite de átomos pesados para QM completo (RHF/STO-3G).\nUsado tanto para o "
              "ligante (não conectado ao tensor hoje) quanto\npara CADA RESÍDUO do sítio no "
              "método 'RHF/STO-3G per residue'\n(um resíduo protonado raramente passa de ~25 "
              "átomos, então o\npadrão de 60 quase sempre é suficiente).",
    },
    "s3_tooltip_vox_batch": {
        "en": "Atoms per batch in Gaussian voxelization.\nCuPy (GPU): 500 recommended "
              "(RTX 8–16 GB).\nNumPy (CPU): 100 recommended.\nHigher values reduce kernel "
              "overhead but increase VRAM usage.\nDetected preset: {batch} ({cupy_status})",
        "pt": "Átomos por lote na voxelização gaussiana.\nCuPy (GPU): 500 recomendado "
              "(RTX 8–16 GB).\nNumPy (CPU): 100 recomendado.\nValores maiores reduzem overhead "
              "de kernel mas aumentam uso de VRAM.\nPreset detectado: {batch} ({cupy_status})",
    },
    "s3_cupy_available": {"en": "CuPy available", "pt": "CuPy disponível"},
    "s3_cupy_unavailable": {"en": "CuPy unavailable", "pt": "CuPy indisponível"},
    "s3_chk_ecfp4": {
        "en": "Use ECFP4 in graph (reduces diversity)",
        "pt": "Usar ECFP4 no grafo (reduz diversidade)",
    },
    "s3_chk_protonate_site": {
        "en": "Protonate site residues (pH-based)", "pt": "Protonar resíduos do sítio (baseado em pH)",
    },
    "s3_tooltip_protonate_site": {
        "en": "Adds explicit H atoms to site residues based on standard pKa\nand the pH below "
              "(Asp/Glu/His/Cys/Tyr/Lys/Arg). Also replaces the\nper-element approximate charge "
              "with real per-atom Gasteiger charges.\nTurning this off reverts to the legacy "
              "behavior (no H, approximate charge).",
        "pt": "Adiciona H explícitos aos resíduos do sítio conforme pKa padrão\ne o pH abaixo "
              "(Asp/Glu/His/Cys/Tyr/Lys/Arg). Também troca a carga\naproximada por elemento por "
              "cargas Gasteiger reais por átomo.\nDesligar volta ao comportamento legado (sem H, "
              "carga aproximada).",
    },
    "s3_tooltip_site_ph": {
        "en": "pH used to decide the protonation state of the site's\nionizable residues "
              "(threshold rule over table pKa values).",
        "pt": "pH usado para decidir o estado de protonação dos resíduos\nionizáveis do sítio "
              "(regra de limiar sobre pKa de tabela).",
    },
    "qm_method_rhf_fragments": {
        "en": "RHF/STO-3G per residue (fragmented)",
        "pt": "RHF/STO-3G por resíduo (fragmentado)",
    },
    "qm_method_gfn_xtb_fragments": {
        "en": "GFN-xTB per residue, parallel (recommended, requires xtb binary)",
        "pt": "GFN-xTB por resíduo, paralelo (recomendado, requer o binário xtb)",
    },
    "qm_method_heuristic": {
        "en": "Heuristic (electronegativity, no QM)", "pt": "Heurística (eletronegatividade, sem QM)",
    },
    "s3_tooltip_site_qm_method": {
        "en": "Level of theory for channels 3-4 (site donor/acceptor field):\n  RHF/STO-3G per "
              "residue — real QM, FMO-style fragmented (slower,\n    ~1s/residue; uses the "
              "already-installed PySCF/GPU4PySCF).\n  GFN-xTB per residue, parallel — "
              "RECOMMENDED if the `xtb` binary is\n    available: semi-empirical, dispatched "
              "across several parallel `xtb`\n    calls (thread pool, threads per call capped so "
              "total threads stay\n    near the CPU count) — validated at ~0.1-0.2s per whole "
              "site.\n  Heuristic — electronegativity + charge, no QM (fastest,\n    also the "
              "automatic fallback when the other one fails).\n\n  A whole-site (unfragmented) "
              "GFN-xTB mode was tried and removed:\n    xtb detects the site as multiple "
              "disconnected molecular fragments\n    (it's a concatenation of independently "
              "capped residues, not a\n    covalently continuous molecule) and segfaults on real "
              "sites, besides\n    taking minutes to converge even when it didn't crash.",
        "pt": "Nível de teoria para os canais 3-4 (campo doador/aceptor do sítio):\n  "
              "RHF/STO-3G por resíduo — QM real, fragmentado tipo FMO (mais lento,\n    "
              "~1s/resíduo; usa PySCF/GPU4PySCF já instalado).\n  GFN-xTB por resíduo, paralelo "
              "— RECOMENDADO se o binário `xtb`\n    estiver disponível: semi-empírico, "
              "distribuído em várias chamadas\n    paralelas de `xtb` (pool de threads, threads "
              "por chamada limitadas\n    para o total ficar perto do número de CPUs) — validado "
              "em\n    ~0,1-0,2s por sítio inteiro.\n  Heurística — eletronegatividade + carga, "
              "sem QM (mais rápido,\n    também o fallback automático quando o outro falha).\n\n"
              "  Um modo de GFN-xTB no sítio inteiro (sem fragmentar) foi\n    testado e removido: "
              "o xtb detecta o sítio como múltiplos fragmentos\n    moleculares desconectados (é "
              "uma concatenação de resíduos capados\n    independentemente, não uma molécula "
              "covalentemente contínua) e\n    crasha com segmentation fault em sítios reais, "
              "além de demorar\n    minutos para convergir mesmo quando não crashava.",
    },
    "s3_lbl_pair": {"en": "Specific pair:", "pt": "Par específico:"},
    "s3_lbl_grid": {"en": "Grid size (N³):", "pt": "Tamanho da grade (N³):"},
    "s3_lbl_padding": {"en": "Padding (Å):", "pt": "Preenchimento (Å):"},
    "s3_lbl_max_qm": {"en": "Max QM atoms:", "pt": "Máx. átomos QM:"},
    "s3_lbl_batch": {"en": "Vox batch size:", "pt": "Tamanho do lote de voxel.:"},
    "s3_lbl_qm_basis": {"en": "QM basis:", "pt": "Base QM:"},
    "s3_tooltip_qm_basis": {
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
    "s3_lbl_site_ph": {"en": "Site pH:", "pt": "pH do sítio:"},
    "s3_lbl_site_qm_method": {
        "en": "Site channels 3-4 method:", "pt": "Método dos canais 3-4 do sítio:",
    },

    # ---------------------------------------------------------------- STEP 4 — Hyperparameter optimization
    "title_step4": {
        "en": "Step 4 — Hyperparameter optimization",
        "pt": "Etapa 4 — Otimização de hiperparâmetros",
    },
    "s4_grp_budget": {"en": "Search budget", "pt": "Orçamento da busca"},
    "s4_grp_space": {"en": "Search space", "pt": "Espaço de busca"},
    "s4_lbl_sampler": {"en": "Search algorithm:", "pt": "Algoritmo de busca:"},
    "s4_tooltip_sampler": {
        "en": "Which configuration to try next. Orthogonal to pruning: "
              "Successive Halving (ASHA) is a PRUNER — it decides when to "
              "kill a running trial, not what to try — and is configured in "
              "the ASHA fields, combining with any algorithm here.",
        "pt": "Qual configuração testar em seguida. Ortogonal à poda: o "
              "Successive Halving (ASHA) é um algoritmo de PODA — decide "
              "quando matar um trial em andamento, não o que testar — e é "
              "configurado nos campos ASHA, combinando com qualquer "
              "algoritmo daqui.",
    },
    "s4_lbl_objective": {"en": "Objective:", "pt": "Objetivo:"},
    "s4_tooltip_objective": {
        "en": "val_recon = reconstruction only, invariant to beta.\n"
              "val_total = recon + beta·KL, only valid with beta FIXED.\n\n"
              "Ticking beta_max or beta_warmup disables val_total: with a "
              "varying beta, val_total rewards whoever drew a smaller beta "
              "rather than the better model.\n\n"
              "val_kl is deliberately absent — its global minimum (KL=0) is "
              "posterior collapse, so it would be a metric whose optimum is "
              "the worst model.",
        "pt": "val_recon = só a reconstrução, invariante a beta.\n"
              "val_total = recon + beta·KL, só válido com beta FIXO.\n\n"
              "Marcar beta_max ou beta_warmup desabilita val_total: com beta "
              "variando, val_total premia quem sorteou um beta menor em vez "
              "do modelo melhor.\n\n"
              "val_kl não é oferecida de propósito — seu mínimo global "
              "(KL=0) é o posterior collapse, ou seja, uma métrica cujo "
              "ótimo é o pior modelo.",
    },
    "s4_note_objective_forced": {
        "en": "val_total disabled: beta is in the search space.",
        "pt": "val_total desabilitado: beta está no espaço de busca.",
    },
    "s4_lbl_phase": {"en": "Phase (dataset):", "pt": "Fase (dataset):"},
    "s4_lbl_n_trials": {"en": "Trials:", "pt": "Trials:"},
    "s4_lbl_trial_epochs": {"en": "Epochs per trial:", "pt": "Épocas por trial:"},
    "s4_lbl_subset": {"en": "Dataset fraction:", "pt": "Fração do dataset:"},
    "s4_lbl_seed": {"en": "Seed:", "pt": "Semente:"},
    "s4_lbl_min_resource": {"en": "ASHA min. epochs:", "pt": "ASHA — épocas mínimas:"},
    "s4_lbl_reduction": {"en": "ASHA reduction factor:", "pt": "ASHA — fator de redução:"},
    "s4_lbl_study": {"en": "Study name:", "pt": "Nome do estudo:"},
    "s4_lbl_device": {"en": "Device:", "pt": "Dispositivo:"},
    "s4_lbl_out_dir": {"en": "Output folder:", "pt": "Pasta de saída:"},
    "s4_lbl_resume": {"en": "Resume from:", "pt": "Retomar de:"},
    "s4_ph_resume": {
        "en": "optional — best_model.pt, to tune fine-tuning (Phases 2/3)",
        "pt": "opcional — best_model.pt, para otimizar fine-tuning (Fases 2/3)",
    },
    "s4_chk_freeze_enc": {
        "en": "Freeze encoder (Phase 3 fine-tuning)",
        "pt": "Congelar encoder (fine-tuning da Fase 3)",
    },
    "s4_chk_keep_ckpts": {
        "en": "Keep checkpoints from every trial (uses much more disk)",
        "pt": "Manter checkpoints de todos os trials (consome muito mais disco)",
    },
    "s4_btn_check_disk": {"en": "Estimate disk usage", "pt": "Estimar espaço em disco"},
    "s4_btn_select_all": {"en": "Select all", "pt": "Marcar todos"},
    "s4_btn_select_default": {"en": "Restore defaults", "pt": "Restaurar padrões"},
    "s4_btn_select_none": {"en": "Clear all", "pt": "Desmarcar todos"},
    "s4_hdr_param": {"en": "Hyperparameter", "pt": "Hiperparâmetro"},
    "s4_hdr_range": {"en": "Range / options", "pt": "Faixa / opções"},
    "s4_prio_high": {"en": "High priority", "pt": "Alta prioridade"},
    "s4_prio_medium": {"en": "Medium priority", "pt": "Média prioridade"},
    "s4_prio_low": {"en": "Low priority", "pt": "Baixa prioridade"},
    "s4_note_space": {
        "en": "Ranges: 'min,max' for continuous/integer parameters; a list of "
              "values for categorical ones. High and medium priority come "
              "pre-selected.",
        "pt": "Faixas: 'mín,máx' para parâmetros contínuos/inteiros; lista de "
              "valores para os categóricos. Alta e média prioridade já vêm "
              "marcadas.",
    },
    "s4_note_objective": {
        "en": "Objective: val_recon whenever beta_max or beta_warmup is in the "
              "search space (val_total = recon + β·KL would reward a smaller β "
              "instead of a better model), val_total otherwise.",
        "pt": "Objetivo: val_recon sempre que beta_max ou beta_warmup estiver "
              "no espaço de busca (val_total = recon + β·KL premiaria um β "
              "menor, não um modelo melhor), val_total caso contrário.",
    },
    # Aviso de fidelidade — trocado ao vivo conforme a fração do dataset e as
    # épocas por trial (ver _opt_update_fidelity_note em CODEEP.py). Com
    # orçamento total a busca deixa de ser proxy e mandar repetir o treino na
    # Etapa 7 seria mandar refazer o que já foi feito.
    "s4_note_screen": {
        "en": "With this budget the search is a PROXY: the ranking comes from "
              "a subset and/or from early epochs. Use it to shortlist 2-3 "
              "configurations, then confirm them with a full training run in "
              "Step 7.",
        "pt": "Com este orçamento a busca é um PROXY: o ranking vem de um "
              "subconjunto e/ou de épocas iniciais. Use-a para selecionar 2-3 "
              "configurações e confirme-as com um treino completo na Etapa 7.",
    },
    "s4_note_near_full": {
        "en": "Full dataset, but a reduced epoch budget: the ranking is "
              "reliable on data and approximate on convergence. Confirm the "
              "top configurations with a full-length run in Step 7.",
        "pt": "Dataset completo, mas orçamento de épocas reduzido: o ranking é "
              "confiável quanto aos dados e aproximado quanto à convergência. "
              "Confirme as melhores com um treino de duração completa na "
              "Etapa 7.",
    },
    "s4_note_full": {
        "en": "With this budget every trial is a COMPLETE training run — this "
              "is a direct comparison, not a screen, and Step 7 need not be "
              "repeated. Beware the cost (trials × epochs) and note that ASHA "
              "still prunes on early epochs. Only the objective gap remains: "
              "compare by the generation metrics of Steps 8-9, not by loss.",
        "pt": "Com este orçamento cada trial é um treino COMPLETO — isto é uma "
              "comparação direta, não uma triagem, e não é preciso repetir a "
              "Etapa 7. Atenção ao custo (trials × épocas) e ao fato de o ASHA "
              "ainda podar nas épocas iniciais. Resta só a lacuna de objetivo: "
              "compare pelas métricas de geração das Etapas 8-9, não pela loss.",
    },
    "s4_dlg_out_dir": {"en": "Optimization output folder", "pt": "Pasta de saída da otimização"},
    "s4_dlg_disk_title": {"en": "Disk space estimate", "pt": "Estimativa de espaço em disco"},
    "s4_dlg_disk_question": {
        "en": "\n\nRun the optimization anyway?",
        "pt": "\n\nExecutar a otimização mesmo assim?",
    },
    "s4_err_no_params": {
        "en": "No hyperparameter selected — tick at least one checkbox to "
              "define the search space.",
        "pt": "Nenhum hiperparâmetro selecionado — marque ao menos um "
              "checkbox para definir o espaço de busca.",
    },
    "s4_err_tensor_not_found": {
        "en": "Tensor dataset not found for Phase {phase}: {tensor_dir}",
        "pt": "Dataset de tensores não encontrado para a Fase {phase}: {tensor_dir}",
    },
    "s4_err_optuna_missing": {
        "en": "Optuna is not installed. Install it with:\n\n"
              "    pip install optuna\n\n"
              "or use Help → Install Requirements.",
        "pt": "O Optuna não está instalado. Instale com:\n\n"
              "    pip install optuna\n\n"
              "ou use Ajuda → Instalar Dependências.",
    },

    # ---------------------------------------------------------------- STEP 5 — Training
    "title_step5": {"en": "Step 5 — CVAE / GNN Training", "pt": "Etapa 5 — Treinamento CVAE / GNN"},
    "s5_lbl_epochs": {"en": "Epochs:", "pt": "Épocas:"},
    "s5_lbl_batch": {"en": "Batch size:", "pt": "Tamanho do lote:"},
    "s5_lbl_latent": {"en": "Latent dim:", "pt": "Dim. latente:"},
    "s5_lbl_hidden": {"en": "Hidden dim:", "pt": "Dim. oculta:"},
    "s5_lbl_gnn_layers": {"en": "GNN layers:", "pt": "Camadas GNN:"},
    "s5_lbl_lr": {"en": "Learning rate:", "pt": "Taxa de aprendizado:"},
    "s5_lbl_beta_max": {"en": "Beta KL (max):", "pt": "Beta KL (máx.):"},
    "s5_lbl_beta_warmup": {"en": "Beta warmup:", "pt": "Beta warmup:"},
    "s5_lbl_pos_weight": {"en": "Position weight:", "pt": "Peso da posição:"},
    "s5_tooltip_pos_weight": {
        "en": "Weight of the 3D position term in the reconstruction loss "
              "(recon = stop_w·L_stop + L_atom + L_bond + L_exist + "
              "pos_w·L_pos + 0.5·L_charge).\n\n"
              "At the default of 2.0 this term alone accounts for about 78% "
              "of recon — it is the model's declared bottleneck. Raising it "
              "buys geometric accuracy at the cost of chemistry; lowering it "
              "does the reverse.",
        "pt": "Peso do termo de posição 3D na perda de reconstrução "
              "(recon = stop_w·L_stop + L_atom + L_bond + L_exist + "
              "pos_w·L_pos + 0,5·L_charge).\n\n"
              "No default de 2,0 esse termo sozinho responde por cerca de "
              "78% da recon — é o gargalo declarado do modelo. Aumentar "
              "compra precisão geométrica ao custo da química; diminuir faz "
              "o inverso.",
    },
    "s5_lbl_stop_weight": {"en": "Stop weight:", "pt": "Peso da parada:"},
    "s5_tooltip_stop_weight": {
        "en": "Weight of the stop head (when to end the ligand construction) "
              "in the reconstruction loss.\n\n"
              "Controls the size distribution of the generated molecules: too "
              "low produces ligands that never stop growing, too high "
              "produces fragments.",
        "pt": "Peso da cabeça de parada (quando encerrar a construção do "
              "ligante) na perda de reconstrução.\n\n"
              "Controla a distribuição de tamanhos das moléculas geradas: "
              "baixo demais gera ligantes que não param de crescer, alto "
              "demais gera fragmentos.",
    },
    "s5_lbl_dropout": {"en": "Dropout:", "pt": "Dropout:"},
    "s5_lbl_weight_decay": {"en": "Weight decay:", "pt": "Weight decay:"},
    "s5_tooltip_weight_decay": {
        "en": "L2 regularization of the AdamW optimizer.\n\n"
              "Raise it if training overfits (validation loss rising while "
              "training loss falls). With the current dataset there is no "
              "sign of overfitting, so the default of 1e-4 is conservative.",
        "pt": "Regularização L2 do otimizador AdamW.\n\n"
              "Aumente se o treino sobreajustar (perda de validação subindo "
              "enquanto a de treino cai). Com o dataset atual não há sinal "
              "de sobreajuste, então o default de 1e-4 é conservador.",
    },
    "s5_lbl_grad_clip": {"en": "Gradient clipping:", "pt": "Recorte do gradiente:"},
    "s5_tooltip_grad_clip": {
        "en": "Maximum gradient norm per step.\n\n"
              "Guards against gradient explosion, which corrupts the weights "
              "with NaN. Only worth changing if the log shows 'Gradiente NaN "
              "detectado' or the loss diverges — lower it in that case.",
        "pt": "Norma máxima do gradiente por passo.\n\n"
              "Protege contra explosão de gradiente, que corrompe os pesos "
              "com NaN. Só vale mexer se o log acusar 'Gradiente NaN "
              "detectado' ou a loss divergir — nesse caso, reduza.",
    },
    "s5_lbl_patience": {"en": "Patience:", "pt": "Paciência:"},
    "s5_lbl_device": {"en": "Device:", "pt": "Dispositivo:"},
    "s5_lbl_resume": {"en": "Resume from:", "pt": "Retomar de:"},
    "s5_lbl_ckpt_out": {"en": "Save ckpt to:", "pt": "Salvar checkpoint em:"},
    "s5_ph_resume": {
        "en": "optional — best_model.pt from previous phase",
        "pt": "opcional — best_model.pt da fase anterior",
    },
    "s5_chk_freeze_enc": {
        "en": "Freeze encoder (Phase 3 fine-tuning)",
        "pt": "Congelar encoder (fine-tuning da Fase 3)",
    },
    "s5_dlg_select_best_model": {"en": "Select best_model.pt", "pt": "Selecione best_model.pt"},
    "s5_err_tensor_not_found": {
        "en": "Tensor dataset not found for Phase {phase}: {tensor_dir}",
        "pt": "Dataset de tensores não encontrado para a Fase {phase}: {tensor_dir}",
    },

    # ---------------------------------------------------------------- STEP 6 — Evaluation
    "title_step6": {
        "en": "Step 6 — Trained Model Evaluation", "pt": "Etapa 6 — Avaliação do Modelo Treinado",
    },
    "s6_chk_do_generate": {
        "en": "Generate ligands during evaluation", "pt": "Gerar ligantes durante a avaliação",
    },
    "s6_lbl_checkpoints": {"en": "Checkpoints:", "pt": "Checkpoints:"},
    "s6_lbl_figures": {"en": "Figures:", "pt": "Figuras:"},
    "s6_btn_training_curves": {"en": "Training Curves", "pt": "Curvas de Treinamento"},
    "s6_btn_performance_report": {"en": "Performance Report", "pt": "Relatório de Desempenho"},
    "s6_lbl_diagnostics": {"en": "Diagnostics:", "pt": "Diagnósticos:"},
    "s6_btn_active_units": {
        "en": "Active Units Diagnostic", "pt": "Diagnóstico de Active Units",
    },
    "s6_tooltip_active_units": {
        "en": "Checks how many of the latent space's dimensions the encoder actually uses "
              "(Burda, Grosse & Salakhutdinov, 2016 — 'active units' metric) — detects PARTIAL "
              "posterior collapse that the aggregate KL value alone can hide (a moderate mean "
              "KL can still mean a handful of active dimensions plus many dead ones). Uses the "
              "checkpoint selected above (Checkpoints field), or checkpoints/phase<N>/ for the "
              "selected phase when left blank.",
        "pt": "Verifica quantas dimensões do espaço latente o encoder realmente usa (métrica "
              "'active units', Burda, Grosse & Salakhutdinov, 2016) — detecta colapso PARCIAL "
              "de posterior que o valor agregado de KL sozinho pode esconder (um KL médio "
              "moderado ainda pode significar poucas dimensões ativas e o resto morto). Usa o "
              "checkpoint selecionado acima (campo Checkpoints), ou checkpoints/phase<N>/ da "
              "fase escolhida quando deixado em branco.",
    },
    "s6_dlg_active_units_title": {
        "en": "Active Units Diagnostic", "pt": "Diagnóstico de Active Units",
    },
    "s6_msg_active_units_no_output": {
        "en": "No output captured — check the checkpoint/tensor paths and try again.",
        "pt": "Nenhuma saída capturada — verifique os caminhos de checkpoint/tensores e "
              "tente novamente.",
    },

    # ---------------------------------------------------------------- STEP 7 — Generation
    "title_step7": {
        "en": "Step 7 — Ligand Candidate Generation", "pt": "Etapa 7 — Geração de Candidatos a Ligante",
    },
    "s7_ph_pdb": {
        "en": "Target protein PDB file (required)", "pt": "Arquivo PDB da proteína-alvo (obrigatório)",
    },
    "s7_ph_site_pdb": {
        "en": "optional — PDB with only binding-site residues",
        "pt": "opcional — PDB apenas com os resíduos do sítio de ligação",
    },
    "s7_ph_out": {
        "en": "JOBS/<job_id>/results/{pdb}/ (default)",
        "pt": "JOBS/<job_id>/results/{pdb}/ (padrão)",
    },
    "s7_ph_ligand_ref": {
        "en": "optional — reference ligand SDF", "pt": "opcional — SDF do ligante de referência",
    },
    "s7_lbl_model_phase": {"en": "Model phase:", "pt": "Fase do modelo:"},
    "s7_lbl_protein_pdb": {"en": "Protein (PDB):", "pt": "Proteína (PDB):"},
    "s7_lbl_site_pdb": {"en": "Site residues (PDB):", "pt": "Resíduos do sítio (PDB):"},
    "s7_lbl_output": {"en": "Output:", "pt": "Saída:"},
    "s7_lbl_ref_ligand": {"en": "Ref. ligand:", "pt": "Ligante de ref.:"},
    "s7_lbl_n_samples": {"en": "No. samples:", "pt": "Nº de amostras:"},
    "s7_lbl_temperature": {"en": "Temperature:", "pt": "Temperatura:"},
    "s7_lbl_cutoff": {"en": "Site cutoff (Å):", "pt": "Cutoff do sítio (Å):"},
    "s7_lbl_top_n": {"en": "Top N report:", "pt": "Relatório Top N:"},
    "s7_lbl_min_atoms": {"en": "Min. site atoms:", "pt": "Átomos mín. do sítio:"},

    # -------------------------------------------- Fechamento com pipeline ativo
    "worker_collect":  {"en": "Collection",   "pt": "Coleta"},
    "worker_curate":   {"en": "Curation",     "pt": "Curadoria"},
    "worker_explore":  {"en": "Exploration",  "pt": "Exploração"},
    "worker_voxelize": {"en": "Voxelization", "pt": "Voxelização"},
    "worker_train":    {"en": "Training",     "pt": "Treinamento"},
    "worker_evaluate": {"en": "Evaluation",   "pt": "Avaliação"},
    "worker_generate": {"en": "Generation",   "pt": "Geração"},
    "worker_active_units": {"en": "Active units diagnostic", "pt": "Diagnóstico de active units"},
    "close_title": {
        "en": "Pipeline still running",
        "pt": "Pipeline ainda em execução",
    },
    "close_text": {
        "en": "The following step is still running: {etapas}.",
        "pt": "A seguinte etapa ainda está em execução: {etapas}.",
    },
    "close_info": {
        "en": "Leaving it in the background keeps the process running, but the "
              "window cannot be reopened — follow the step's log file instead. "
              "For training, that is training.log inside the checkpoint folder. "
              "CODEEP will exit on its own once the step finishes.",
        "pt": "Deixar em segundo plano mantém o processo rodando, mas a janela "
              "não poderá ser reaberta — acompanhe pelo arquivo de log da etapa. "
              "No treinamento, é o training.log dentro da pasta do checkpoint. "
              "O CODEEP encerra sozinho quando a etapa terminar.",
    },
    "close_btn_background": {
        "en": "Leave running in background",
        "pt": "Deixar rodando em segundo plano",
    },
    "close_btn_stop": {
        "en": "Stop and exit",
        "pt": "Interromper e sair",
    },
    "close_btn_cancel": {
        "en": "Cancel",
        "pt": "Cancelar",
    },

    # ------------------------------------------- Definição do sítio sem ligante
    "s7_chk_use_site_center": {
        "en": "Define site by center coordinates",
        "pt": "Definir sítio por coordenadas do centro",
    },
    "s7_lbl_site_center": {"en": "Site center:", "pt": "Centro do sítio:"},
    "s7_lbl_site_radius": {"en": "Site radius (Å):", "pt": "Raio do sítio (Å):"},
    "s7_chk_heuristic_search": {
        "en": "Heuristic search (P2Rank)",
        "pt": "Busca heurística (P2Rank)",
    },
    "s7_lbl_pocket_rank": {"en": "Pocket to use:", "pt": "Bolso a usar:"},
    "s7_tooltip_site_center": {
        "en": "Use when the protein has no co-crystallized ligand. Residues with at "
              "least one atom within the radius of this point define the site. The "
              "coordinates must be in the same frame as the supplied PDB.",
        "pt": "Use quando a proteína não tem ligante co-cristalizado. Os resíduos com "
              "ao menos um átomo dentro do raio em torno deste ponto definem o sítio. "
              "As coordenadas devem estar no mesmo referencial do PDB fornecido.",
    },
    "s7_tooltip_site_radius": {
        "en": "Default 10 Å. Larger than the 6 Å used when a ligand is present: there "
              "the distance starts at the ligand atoms, which already fill the cavity, "
              "whereas here it starts at a single central point.",
        "pt": "Padrão 10 Å. Maior que os 6 Å usados quando há ligante: naquele caso a "
              "distância parte dos átomos do ligante, que já preenchem a cavidade, "
              "enquanto aqui parte de um único ponto central.",
    },
    "s7_tooltip_heuristic_search": {
        "en": "Predicts binding pockets with P2Rank (Krivák & Hoksza, 2018) and uses "
              "the chosen one, ranked by ligandability. Disables the manual "
              "coordinates. Requires P2Rank installed and Java 17+; the full ranking "
              "is written to pockets_p2rank.json in the output folder.",
        "pt": "Prediz bolsos com o P2Rank (Krivák & Hoksza, 2018) e usa o bolso "
              "escolhido, na ordem de ligandability. Desabilita as coordenadas "
              "manuais. Exige P2Rank instalado e Java 17+; o ranking completo é "
              "gravado em pockets_p2rank.json na pasta de saída.",
    },
    "s7_tooltip_pocket_rank": {
        "en": "Which pocket to use from the P2Rank ranking (1 = highest score). Check "
              "pockets_p2rank.json from a previous run to pick another one.",
        "pt": "Qual bolso usar no ranking do P2Rank (1 = maior score). Consulte o "
              "pockets_p2rank.json de uma execução anterior para escolher outro.",
    },
    "s7_err_site_conflict": {
        "en": "The site PDB already delimits the pocket. Clear it before using center "
              "coordinates or heuristic search.",
        "pt": "O PDB do sítio já delimita o bolso. Remova-o antes de usar coordenadas "
              "do centro ou busca heurística.",
    },
    "s7_err_center_zero": {
        "en": "Set the site center coordinates (they are all zero).",
        "pt": "Informe as coordenadas do centro do sítio (estão todas em zero).",
    },
    "s7_chk_protonate_site": {
        "en": "Protonate site residues (pH-based)", "pt": "Protonar resíduos do sítio (baseado em pH)",
    },
    "s7_tooltip_gen_qm_method": {
        "en": "Fallback only: when the selected checkpoint has its training voxelization "
              "config saved (checkpoints trained after this feature was added), that value is "
              "always used instead, with a log warning if it differs from this field — this "
              "choice only takes effect for checkpoints trained before that (no config saved).",
        "pt": "Só como reserva: quando o checkpoint escolhido tem a config de voxelização do "
              "treino salva (checkpoints treinados depois desta funcionalidade existir), esse "
              "valor é sempre usado no lugar deste, com aviso no log se divergir — esta escolha "
              "só tem efeito para checkpoints treinados antes disso (sem config salva).",
    },
    "s7_tooltip_gen_qm_basis": {
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
    "s7_lbl_final_ff": {"en": "Final geometry FF:", "pt": "FF de geometria final:"},
    "s7_tooltip_final_ff": {
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
    "s7_chk_induced_fit": {
        "en": "Induced-fit (flexible side chains)", "pt": "Induced-fit (cadeias laterais flexíveis)",
    },
    "s7_tooltip_induced_fit": {
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
    "s7_err_invalid_pdb": {
        "en": "Please select a valid PDB file for the target protein.",
        "pt": "Selecione um arquivo PDB válido para a proteína-alvo.",
    },
    "s7_err_invalid_site_pdb": {
        "en": "Please select a valid PDB file containing only the binding-site residues.",
        "pt": "Selecione um arquivo PDB válido contendo apenas os resíduos do sítio de ligação.",
    },
    "s7_err_no_checkpoint": {
        "en": "No trained checkpoint found for Phase {phase}.",
        "pt": "Nenhum checkpoint treinado encontrado para a Fase {phase}.",
    },
    "s7_dlg_select_pdb": {"en": "Select PDB file", "pt": "Selecione o arquivo PDB"},
    "s7_dlg_select_site_pdb": {
        "en": "Select binding-site residues PDB",
        "pt": "Selecione o PDB dos resíduos do sítio de ligação",
    },
    "s7_dlg_output_folder": {"en": "Output folder", "pt": "Pasta de saída"},
    "s7_dlg_reference_ligand": {"en": "Reference ligand", "pt": "Ligante de referência"},

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
    "msg_run_dataexp_first": {
        "en": "Run Data Exploration (Step 2) with Phase {phase} first.\n\nExpected at:\n{fig_dir}",
        "pt": "Execute Exploração de Dados (Etapa 2) com a Fase {phase} primeiro.\n\n"
              "Esperado em:\n{fig_dir}",
    },
    "msg_fig_title_with_phase": {
        "en": "{title} — Phase {phase}", "pt": "{title} — Fase {phase}",
    },
    "msg_run_step6_first": {
        "en": "Run Step 6 with Phase {phase} first.\n\nExpected at:\n{fig_dir}",
        "pt": "Execute a Etapa 6 com a Fase {phase} primeiro.\n\nEsperado em:\n{fig_dir}",
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
