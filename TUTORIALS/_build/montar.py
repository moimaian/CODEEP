# -*- coding: utf-8 -*-
"""Monta o manual (.docx e .pdf). Uso:  python3 montar.py <pasta_das_figuras> <pasta_de_saida> [pt|en]

Duas passagens: a primeira gera o PDF para descobrir em que página cai cada título; a segunda grava o
sumário com esses números. O sumário é um campo TOC do Word, já preenchido, e pode ser atualizado ali.
"""
import os, re, subprocess, sys, json, importlib, shutil
aqui = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, aqui)
fig = os.path.abspath(sys.argv[1]); saida = os.path.abspath(sys.argv[2])
LANG = sys.argv[3] if len(sys.argv) > 3 else "pt"
os.environ["CODEEP_FIG"] = fig
import docxlib
conteudo_mod = importlib.import_module("manual_pt_BR" if LANG == "pt" else "manual_en_US")
from docx.shared import Pt, RGBColor

sys.path.insert(0, os.path.join(aqui, "..", ".."))
VERSAO = re.search(r'VERSAO\s*=\s*"([^"]+)"', open(os.path.join(aqui, "..", "..", "versao.py")).read()).group(1)
LOGO = os.path.join(aqui, "..", "..", "ICONS", "CODEEP.png")
NOME = "manual_do_usuario_CODEEP_pt_BR" if LANG == "pt" else "User_manual_CODEEP_en-US"
TXT_SUMARIO = "Sumário" if LANG == "pt" else "Contents"

def construir(paginas, estrutura):
    m = docxlib.Manual(VERSAO, paginas, LANG)
    cap = ({"pt": ("MANUAL DO USUÁRIO", "Plataforma de Geração de Ligantes de Novo",
                   "Aprendizado Profundo Condicionado ao Sítio de Ligação", f"Versão {VERSAO}",
                   "CEB/UFPR · Licença GPL-3.0-or-later"),
            "en": ("USER MANUAL", "De Novo Ligand Generation Platform",
                   "Deep Learning Conditioned on the Binding Site", f"Version {VERSAO}",
                   "CEB/UFPR · GPL-3.0-or-later License")})[LANG]
    m.capa(LOGO, [
        (cap[0], 20, True, docxlib.COR, 26),
        ("CODEEP", 40, True, None, 4),
        (cap[1], 14, False, docxlib.CINZA, 14),
        (cap[2], 11, False, docxlib.CINZA, 2),
        (cap[3], 11, False, docxlib.CINZA, 90),
        (cap[4], 10, False, docxlib.CINZA, 2),
        ("https://github.com/moimaian/CODEEP", 10, False, docxlib.CINZA, 2)])
    m.sumario(estrutura)
    n_antes = len(m.titulos)
    conteudo_mod.conteudo(m)
    return m

def para_pdf(docx):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", os.path.dirname(docx), docx],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def paginas_dos_titulos(pdf, titulos):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    res, ini = {}, 0
    textos = []
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-layout", pdf, "-"],
                           capture_output=True, text=True).stdout
        textos.append(t)
    # a página do sumário não conta: procura só depois dela
    inicio = next(i for i, t in enumerate(textos) if TXT_SUMARIO in t) + 1
    while inicio < n and "...." in textos[inicio]:      # sumário que ocupa mais de uma página
        inicio += 1
    cur = inicio
    for _, tit in titulos:
        for p in range(cur, n):
            if any(l.strip().startswith(tit) for l in textos[p].splitlines()):
                res[tit] = p + 1; cur = p; break
    return res

# passagem 0: só coleta a estrutura de títulos
tmp = docxlib.Manual(VERSAO, None, LANG); conteudo_mod.conteudo(tmp); estrutura = list(tmp.titulos)
# passagem 1: sumário com números provisórios
os.makedirs(saida, exist_ok=True)
work = os.path.join(saida, "_tmp"); os.makedirs(work, exist_ok=True)
d1 = os.path.join(work, NOME + ".docx")
construir({}, estrutura).salvar(d1); para_pdf(d1)
pg = paginas_dos_titulos(os.path.join(work, NOME + ".pdf"), estrutura)
faltam = [t for _, t in estrutura if t not in pg]
print("títulos sem página:", faltam)
# passagem 2: número reais
d2 = os.path.join(saida, NOME + ".docx")
construir(pg, estrutura).salvar(d2); para_pdf(d2)
shutil.rmtree(work)
print("gerado:", d2, "| páginas:", re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", d2[:-5] + ".pdf"], capture_output=True, text=True).stdout).group(1))
