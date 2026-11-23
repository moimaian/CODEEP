#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fonte unica da versao do CODEEP.

VERSAO segue o esquema MAJOR.MINOR.PATCH-REVISAO, onde:
- MAJOR sobe quando ha mudanca incompatível de fluxo, estrutura de projeto ou comportamento.
- MINOR sobe quando ha funcionalidade nova visivel ao usuario, novas opcoes ou novo fluxo.
- PATCH sobe para correcoes de bug e ajustes menores sem funcionalidade nova.
- REVISAO sobe apenas para ajustes de empacotamento/distribuicao sem mudanca de codigo.

Exemplos praticos no CODEEP:
- nova persistencia de projetos, novas abas, novos relatorios, novos grupos de interface -> MINOR
- correcao de callback, validacao, texto, layout ou comportamento ja existente -> PATCH
- mudanca que invalida compatibilidade com projetos/rotinas anteriores -> MAJOR
"""

APP_NOME = "CODEEP"
VERSAO = "1.2.1-1"
LICENCA_CURTA = "GPL-3.0-or-later"
