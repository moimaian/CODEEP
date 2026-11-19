# Versionamento do CODEEP

A versao do CODEEP segue o padrao `MAJOR.MINOR.PATCH-REVISAO`, no mesmo espirito
adotado no GitFace:

- `MAJOR.MINOR.PATCH` e a versao do programa, seguindo SemVer.
- `-REVISAO` e a revisao de empacotamento/distribuicao. Ela sobe apenas quando a
  mudanca nao altera o codigo do programa em si.

Exemplo:

```text
1.0.0    -1
└─┬─┘    └┬┘
  │       └─ revisao de empacotamento/distribuicao
  └─ versao do programa
```

## Fonte unica da versao

A fonte unica do numero e [versao.py](../versao.py):

```python
VERSAO = "1.0.0-1"
```

Esse valor e o mesmo que:

- aparece no rodape da Home do CODEEP;
- alimenta o campo `Version=` do atalho `.desktop` gerado pelo programa;
- deve ser reutilizado por futuros scripts de build/empacotamento do projeto.

## Como decidir qual numero sobe

- `PATCH`: corrigiu bug sem alterar o comportamento esperado.
- `MINOR`: adicionou funcionalidade nova sem quebrar compatibilidade.
- `MAJOR`: mudou ou removeu comportamento existente de forma que o usuario precise se adaptar.

## Quando subir apenas a revisao

Mantenha `MAJOR.MINOR.PATCH` e suba apenas `-REVISAO` quando a mudanca for de
empacotamento ou distribuicao, sem alterar o codigo funcional do programa.

Exemplos:

- ajustar launcher `.desktop`;
- mudar arquivos em `packaging/`;
- corrigir instrucoes de instalacao ou scripts de empacotamento.

## Fluxo recomendado para um novo release

1. Decida o proximo `MAJOR.MINOR.PATCH`.
2. Atualize `VERSAO` em [versao.py](../versao.py).
3. Se a mudanca for so de empacotamento, mantenha `MAJOR.MINOR.PATCH` e suba apenas `-REVISAO`.
4. Use esse mesmo numero em qualquer changelog, script de build ou pacote futuro do CODEEP.

## Ao gerar um novo `.deb`

1. Decida o próximo `MAJOR.MINOR.PATCH`: a pergunta não é "quanto código mudou",
   e sim *alguém que já usa o CODEEP vai precisar fazer algo diferente, ou algo
   deixou de funcionar, por causa desta mudança?* — `PATCH` para correção de bug,
   `MINOR` para funcionalidade nova sem quebrar nada, `MAJOR` para mudança que
   exige adaptação (por exemplo, formato dos arquivos de estado do projeto ou
   renumeração/remoção de uma etapa do fluxo sem migração automática).
2. Atualize `VERSAO` em [versao.py](../versao.py).
3. Adicione uma entrada no topo de [packaging/deb/changelog](deb/changelog)
   (formato Debian: `codeep (X.Y.Z-1) unstable; urgency=low`, lista do que
   mudou, linha ` -- Nome <email>  data`) com o **mesmo** número do passo 2.
4. Rode `packaging/deb/build.sh` (sem argumento, usa o número de `versao.py`).
   O pacote sai em `packaging/dist/codeep_X.Y.Z-1_all.deb`.

O `build.sh` recusa rodar se `versao.py` e o topo do changelog divergirem. Toda
mudança de código reseta a revisão para `-1`; mudança só em `packaging/` mantém o
`X.Y.Z` e sobe o `-N`. Detalhes do pacote em [deb/README.md](deb/README.md).
