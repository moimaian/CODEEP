# -*- coding: utf-8 -*-
"""Gera as capturas de tela usadas no manual (GUI em português, sem abrir janela).

Uso:  python capturas.py <pasta_de_saida> [pt|en]
Rode com o Python do ambiente do CODEEP (~/.venv/CODEEP/bin/python -I), a partir de qualquer pasta.
Também copia as figuras reais de saída do pipeline (curvas, relatório, painel, grade de candidatos).
"""
import os, sys, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, RAIZ); os.chdir(RAIZ)
OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
LANG = sys.argv[2] if len(sys.argv) > 2 else "pt"

from PyQt5.QtWidgets import QApplication, QPushButton, QScrollArea, QDialog
from PyQt5.QtCore import QTimer
import CODEEP

app = QApplication([]); w = CODEEP.MainWindow()
w.resize(1400, 1300); w.show()
w._definir_idioma(LANG); w._altura_util_tela = lambda: 3000
def vazio(n=6):
    for _ in range(n): app.processEvents()
vazio()

abas = {0: "inicio", 1: "config", 2: "etapa1", 3: "etapa2", 4: "etapa3", 5: "etapa4", 6: "etapa5", 7: "etapa6"}
for i, nome in abas.items():
    w.tabs.setCurrentIndex(i); pag = w.tabs.widget(i)
    for b in pag.findChildren(QPushButton):          # recolhe log/curvas: o painel de parâmetros ocupa a aba
        if b.isCheckable() and b.isChecked() and ("Log" in b.text() or "Curv" in b.text()):
            b.setChecked(False)
    w.resize(1400, 1300); vazio(8)
    for _ in range(3):                                # ajusta a altura da janela ao conteúdo
        sh = max(pag.minimumSizeHint().height(),
                 sum(sc.widget().sizeHint().height() - sc.viewport().height()
                     for sc in pag.findChildren(QScrollArea)[:1] if sc.widget()) + pag.height())
        w.resize(1400, w.height() - pag.height() + sh + 6); vazio(8)
    if i == 2: w.resize(1400, w.height() + 70); vazio(8)                         # log compartilhado
    if i == 5: w.resize(1400, w.height() - pag.height() + 945); vazio(8)         # espaço elástico
    pag.grab().save(f"{OUT}/{nome}.png"); print(nome)

from CODEEP_requirements import RequirementsInstaller
r = RequirementsInstaller(None); r.resize(1100, 760); r.show(); vazio(8)
r.grab().save(f"{OUT}/dependencias.png"); r.close(); print("dependencias")

# relatório REAL do diagnóstico (roda o código do programa sobre o checkpoint da Fase 1)
import io, contextlib, tempfile
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    CODEEP._pipeline_check_active_units(f"{RAIZ}/checkpoints/phase1", f"{RAIZ}/data/tensors",
                                        log_dir=tempfile.mkdtemp(), idioma=LANG)
texto = "\n".join(l for l in buf.getvalue().splitlines() if not l.startswith("LOGFILE:")).strip()
def tira():
    d = [x for x in app.topLevelWidgets() if isinstance(x, QDialog) and x.isVisible()][0]
    d.resize(760, 330); vazio()
    d.grab().save(f"{OUT}/active_units.png"); d.accept()
QTimer.singleShot(400, tira)
w._show_active_units_dialog(CODEEP.i18n.t("s6_dlg_active_units_title", LANG), texto,
                            f"{RAIZ}/data/logs/active_units.log", True)
print("active_units")

# figuras reais produzidas pelo pipeline
for orig, dest in (("data/figures/training_curves.png", "training_curves.png"),
                   ("data/figures/performance_report.png", "performance_report.png"),
                   ("data/figures/summary_dashboard.png", "summary_dashboard.png"),
                   ("PROJECTS/2026-11-14_00-49_TESTE_ARCH5/results/2pc8/top20_candidates.png", "top20.png")):
    if os.path.isfile(os.path.join(RAIZ, orig)):
        shutil.copy2(os.path.join(RAIZ, orig), os.path.join(OUT, dest)); print(dest)
    else:
        print("AUSENTE:", orig)
