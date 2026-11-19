# Empacotamento .deb do CODEEP

## Build

```bash
packaging/deb/build.sh              # versão = versao.py (ex.: 1.1.0-1)
packaging/deb/build.sh 1.1.0-2      # versão explícita
```

Gera `packaging/dist/codeep_<versão>_all.deb`. Antes de gerar um novo pacote,
atualize `versao.py` e o topo de `packaging/deb/changelog` com o mesmo número —
o script recusa rodar se os dois divergirem. Regras para escolher o número
(MAJOR/MINOR/PATCH-N) em [`packaging/VERSIONING.md`](../VERSIONING.md).

Dependências para *buildar* (não para instalar/usar): `dpkg-deb`, `rsync`
(já vêm em qualquer Debian/Ubuntu/Mint). `lintian` é opcional.

## O que vai no pacote — e o que não vai

Vão para `/opt/codeep`: `CODEEP.py`, `CODEEP_requirements.py`, `versao.py`,
`MODULES/`, `ICONS/` e `TUTORIALS/` (só os manuais; a subpasta `_build/` fica de
fora), mais `LICENSE.txt` e `README.md` da raiz.

**Ficam de fora, de propósito:**

- `data/`, `checkpoints/`, `hpo/`, `PROJECTS/` — dados, modelos e resultados do
  usuário. São gigabytes e, principalmente, o launcher sincroniza
  `/opt/codeep` sobre `~/CODEEP`: um modelo treinado empacotado sobrescreveria o
  do usuário;
- `tools/` — `xtb` e P2Rank, que a janela *Install Requirements* baixa para
  `~/CODEEP/tools`;
- `TEST/`, `results/`, `ARTIGO/` e demais materiais de desenvolvimento.

## Instalar

```bash
sudo apt install ./packaging/dist/codeep_<versão>_all.deb
```

Isso cria:
- `/opt/codeep` — código da aplicação (arquivo do pacote, gerenciado pelo dpkg);
- `/usr/bin/codeep` — launcher;
- `/usr/share/applications/CODEEP.desktop` — entrada no menu.

## O que acontece no primeiro clique em "CODEEP" no menu

1. `/usr/bin/codeep` sincroniza `/opt/codeep` → `$HOME/CODEEP` (sem tocar em
   `PROJECTS/`, `data/`, `checkpoints/`, `hpo/` nem `tools/`) e chama
   `python3 $HOME/CODEEP/CODEEP.py`.
2. `CODEEP.py` (código já existente, inalterado por este empacotamento) detecta
   que ainda não está rodando no venv-alvo e, em `_bootstrap_venv`:
   - pede confirmação no terminal, cria `$HOME/.venv/CODEEP`, instala o PyQt5 e
     se reinicia (`os.execve`) já dentro do venv;
   - o restante das dependências (PyTorch, RDKit etc.) é instalado pela janela
     *Install Requirements*, aberta pela aba INÍCIO ou por *Help > Install
     Requirements*, e o PyTorch é escolhido conforme a GPU detectada;
   - grava o atalho `~/.local/share/applications/CODEEP.desktop`.

O `.desktop` do pacote usa `Terminal=false`: o CODEEP abre sem terminal, e a
saída do programa é gravada em `~/.local/state/CODEEP/terminal.log`
(*View Terminal*, na aba INÍCIO). A exceção é a primeira execução: sem o venv
criado, o passo 2 usa `input()` para confirmar, então `/usr/bin/codeep` se
reabre sozinho dentro de um emulador de terminal (gnome-terminal,
x-terminal-emulator ou xterm) só nessa vez. Nas execuções seguintes o launcher
entra direto no Python do venv, com o mesmo comando do atalho que o próprio
programa grava.

## Quem desenvolve o CODEEP: cuidado com a sincronização

O launcher copia `/opt/codeep` sobre `~/CODEEP`. Se `~/CODEEP` for a sua cópia
de trabalho do repositório, isso sobrescreveria alterações ainda não comitadas.
Por isso o launcher **não sincroniza quando `~/CODEEP` contém uma pasta `.git`**
(avisa no stderr e segue abrindo o programa). Para testar a instalação como um
usuário comum, use outra conta ou uma máquina sem o repositório.

## Por que o pacote não cria o venv/instala as dependências no `postinst`

`postinst`/`postrm` rodam como **root**, durante `apt install`/`dpkg -i`.
Nesse momento:

- `$HOME` não é o home do usuário final (é `/root` ou indefinido) — não dá para
  localizar `$HOME/CODEEP` de forma confiável, e em máquina multiusuário a
  pergunta "de qual usuário?" nem faz sentido para um pacote de sistema;
- instalar via pip pacotes pesados (PyTorch, RDKit, CuPy) depende de rede,
  demora minutos, e se falhar no meio deixa o `dpkg` em estado
  `half-configured`, travando qualquer `apt` seguinte até conserto manual —
  contra as boas práticas de empacotamento Debian.

Por isso o pacote fica deliberadamente fino: só coloca arquivos em
`/opt/codeep`, registra o menu e deixa o bootstrap por-usuário para o próprio
`CODEEP.py`.

## Atalho do menu: um só, sempre pelo launcher

O pacote instala `CODEEP.desktop` em `/usr/share/applications` e o programa grava
um `CODEEP.desktop` em `~/.local/share/applications` — **o mesmo nome**, o que
faz o do usuário prevalecer sobre o do sistema e o menu mostrar uma única
entrada "CODEEP" (nomes diferentes, como `codeep.desktop` e `CODEEP.desktop`,
apareceriam duas vezes).

Só trocar o nome, porém, abriria um furo: o atalho gravado pelo programa abria
`~/CODEEP/CODEEP.py` direto, sem passar pelo launcher, e as atualizações do
`apt upgrade` deixariam de chegar em `~/CODEEP`. Por isso, em
`_desktop_content()` (`CODEEP.py`), quando `/usr/bin/codeep` existe o atalho do
usuário passa a usar `Exec=/usr/bin/codeep`. Sem o pacote (execução a partir do
código-fonte), continua abrindo o Python do venv diretamente.

## Atualizações

`apt upgrade` atualiza `/opt/codeep`. Na próxima vez que o usuário abrir o
CODEEP pelo menu, o launcher sincroniza a versão nova para `$HOME/CODEEP`. O
venv em `~/.venv/CODEEP` não é recriado por uma atualização do pacote; para
reinstalar dependências científicas use a janela *Install Requirements*.

## Desinstalar

```bash
sudo apt remove codeep     # mantém ~/CODEEP e ~/.venv/CODEEP
sudo apt purge codeep      # idem — avisa no terminal, mas não apaga nada em $HOME
```

Dados por-usuário (`~/CODEEP`, incluindo `data/`, `checkpoints/` e `PROJECTS/`,
e `~/.venv/CODEEP`) nunca são apagados automaticamente pelo pacote.

## Diferenças em relação ao empacotamento do CODRUG

- sem época `1:` no `Version` — o CODEEP nunca teve `.deb` numerado por data,
  então não há o que sobrepor (uma época pode ser acrescentada depois, se um dia
  for preciso);
- a lista de arquivos copiados é a do CODEEP (ver acima) e o launcher preserva
  `data/`, `checkpoints/`, `hpo/` e `tools/`, que o CODRUG não tem;
- o launcher não sincroniza sobre um `~/CODEEP` que seja repositório git;
- a fonte única da versão é `versao.py` (na raiz), e não `BIN/versao.py`;
- o atalho gravado pelo programa em `~/.local` aponta para o launcher quando o pacote está instalado
  (ver acima).
