# Como regenerar os manuais / How to regenerate the manuals

Os manuais (`../manual_do_usuario_CODEEP_pt_BR` e `../User_manual_CODEEP_en-US`, em `.docx` e `.pdf`)
são gerados por scripts, para poderem ser refeitos quando a interface mudar. Requer `python3` com
`python-docx`, `soffice` (LibreOffice) e `pdftotext`.

```bash
# 1) capturas de tela da GUI no idioma desejado (usa o Python do ambiente do CODEEP)
~/.venv/CODEEP/bin/python -I capturas.py /tmp/figuras_pt pt
~/.venv/CODEEP/bin/python -I capturas.py /tmp/figuras_en en

# 2) monta o .docx e o .pdf (sumário com os números de página reais)
python3 montar.py /tmp/figuras_pt .. pt
python3 montar.py /tmp/figuras_en .. en
```

- `manual_pt_BR.py` / `manual_en_US.py` — o texto de cada versão (mantenha as duas em sincronia);
- `docxlib.py` — estilos e blocos (títulos, tabelas, figuras, notas);
- `capturas.py` — capturas das abas e das figuras de saída do pipeline;
- `montar.py` — gera o documento em duas passagens (a primeira descobre as páginas dos títulos).

Os rótulos citados no texto seguem a interface na versão registrada em `versao.py`.
