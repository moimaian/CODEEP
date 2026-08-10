# Benchmark do CODEEP sobre CrossDocked2020

## Objetivo

Este documento define um protocolo de benchmark para comparar o CODEEP com outras plataformas de geracao de ligantes 3D condicionada ao sitio de ligacao. O foco e comparar metodos sob um mesmo conjunto de alvos, um mesmo regime de amostragem e uma mesma bateria de metricas quimicas, geometricas e funcionais.

O protocolo foi desenhado para ficar alinhado com trabalhos do tipo do liGAN e similares, que usam bases estruturais receptor-ligante com avaliacao por qualidade molecular, diversidade, plausibilidade 3D e rescoring por docking.

## Escopo

O benchmark deve responder a quatro perguntas:

1. O CODEEP gera moleculas quimicamente validas, diversas e nao triviais?
2. As moleculas geradas sao geometricamente plausiveis no contexto do bolso?
3. O ranking interno do CODEEP se sustenta quando comparado por uma metrica externa comum a todos os metodos?
4. O modelo responde ao receptor ou produz quase a mesma distribuicao independentemente do bolso?

## Benchmark principal

### Base recomendada

Usar o CrossDocked2020 como benchmark principal.

Motivos:

- E uma base publica orientada a estrutura receptor-ligante 3D.
- Ja foi usada por trabalhos de geracao condicionada ao receptor.
- Permite split por bolso ou cluster estrutural, reduzindo leakage.
- E adequada para avaliacao por docking/rescoring uniforme.

### Unidade de avaliacao

A unidade de avaliacao deve ser o alvo estrutural, nao a molecula individual isolada.

Para cada alvo:

- um receptor ou bolso definido
- um ligante de referencia
- um conjunto fixo de amostras geradas por metodo
- o mesmo protocolo de minimizacao, filtros e rescoring

### Split recomendado

Usar split por bolso ou cluster estrutural. Nunca usar split aleatorio de pares receptor-ligante.

Regra recomendada:

- treino: clusters de bolsos exclusivos
- validacao: clusters de bolsos exclusivos
- teste: clusters de bolsos exclusivos

Se a plataforma comparada nao tiver sido treinada no CrossDocked2020, ainda assim a avaliacao deve ser feita sobre um subconjunto comum de alvos do split de teste.

## Protocolo completo

### Fase 1: definicao do subconjunto de teste

Selecionar alvos do split de teste com estes criterios:

1. ao menos 5 ligantes unicos associados ao bolso
2. receptor com sitio razoavelmente bem definido
3. ligantes organicos pequenos, sem depender de cofatores exoticos como unico determinante do encaixe
4. diversidade de familias de alvo para evitar benchmark enviesado para um tipo unico de bolso
5. preferencia por alvos para os quais seja possivel calcular docking/rescoring de forma estavel

### Fase 2: normalizacao de entrada

Cada metodo deve receber, para cada alvo, a mesma definicao minima de contexto estrutural:

1. receptor completo ou bolso recortado segundo uma regra fixa
2. ligante de referencia apenas quando o metodo precisar desse artefato para condicionar ou avaliar similaridade
3. mesma regra de protonacao, quando possivel
4. mesma regra de recorte do sitio, por exemplo cutoff em Angstrom a partir do ligante de referencia

Para o CODEEP, o ideal e fixar explicitamente:

- cutoff do sitio
- uso ou nao de protonacao do sitio
- campo de forca final
- modo induced-fit ou rigido

### Fase 3: geracao por metodo

Para cada alvo e para cada plataforma comparada:

1. gerar exatamente `N` candidatos brutos, por exemplo `N = 100`
2. usar o mesmo numero de tentativas por alvo em todos os metodos
3. registrar quantos candidatos falharam antes de qualquer filtro posterior
4. deduplicar por SMILES canonico para metricas de unicidade, mas manter o bruto para metricas de eficiencia

Parametros recomendados para o benchmark principal:

- 100 amostras por alvo
- 3 repeticoes por alvo com seeds diferentes
- mesma politica de temperatura ou diversidade dentro de cada metodo, documentada e fixa

### Fase 4: filtros e pos-processamento comuns

Aplicar a mesma bateria de pos-processamento a todos os metodos antes da comparacao final:

1. sanitizacao RDKit
2. remocao de fragmentos desconexos, se a politica do benchmark exigir molecula unica conectada
3. minimizacao geometrica local
4. checagem de clash no bolso
5. rescoring externo por ferramenta comum

O benchmark deve separar claramente:

- metrica antes de filtros
- metrica apos filtros quimicos
- metrica apos filtros geometricos
- metrica apos rescoring

### Fase 5: metricas primarias

#### A. Qualidade quimica

Medir por alvo e agregado:

- validity
- uniqueness
- novelty contra treino
- internal diversity
- QED medio
- SA score medio
- MW medio
- LogP medio
- fracao que passa Lipinski
- fracao que passa Veber

#### B. Qualidade estrutural e plausibilidade 3D

Medir por alvo e agregado:

- fracao sem clash severo com o bolso
- numero medio de clashes por candidato
- distancia minima de clash
- conformidade de comprimento de ligacao
- RMSD apos minimizacao, quando houver referencia apropriada
- estabilidade pos-minimizacao

Se possivel, incorporar uma bateria tipo PoseBusters ou equivalente.

#### C. Compatibilidade funcional com o receptor

Medir com uma ferramenta externa comum a todos os metodos:

- score Vina minimizado
- score gnina
- afinidade prevista por CNN, se gnina estiver disponivel
- fracao de candidatos que superam o ligante de referencia no mesmo score
- melhor score por alvo
- media dos top-10 por alvo

#### D. Eficiencia

Medir:

- tempo por alvo
- tempo por candidato valido
- taxa de sucesso bruto
- uso de GPU e CPU, se relevante

### Fase 6: metricas secundarias de condicionalidade

Para testar se o metodo realmente responde ao receptor:

1. escolher um subconjunto pequeno de alvos
2. criar perturbacoes controladas no bolso, por exemplo mutacao de residuos-chave ou mudanca sistematica no recorte do sitio
3. gerar novamente candidatos sob o receptor modificado
4. comparar distribuicoes de fingerprint similarity, docking score e propriedades fisico-quimicas

Se o metodo for realmente receptor-condicionado, a distribuicao das moleculas geradas deve mudar de forma consistente com a perturbacao do bolso.

### Fase 7: agregacao e estatistica

Reportar sempre por alvo e agregado.

Agregados minimos:

- media
- mediana
- desvio padrao
- intervalo interquartil

Para comparar metodos:

- usar teste pareado por alvo
- preferir Wilcoxon signed-rank para metricas nao gaussianas
- usar bootstrap para IC de metricas agregadas
- corrigir multiplas comparacoes quando necessario

### Fase 8: artefatos de saida

Cada rodada de benchmark deve produzir:

1. um CSV por alvo com todos os candidatos
2. um CSV agregado por metodo
3. um JSON com configuracao completa da rodada
4. um relatorio resumido com tabelas e graficos
5. um conjunto opcional de SDFs dos top candidatos

Schema minimo por candidato:

- target_id
- run_seed
- method_name
- raw_index
- smiles
- valid
- unique
- novel
- qed
- sa_score
- mw
- logp
- lipinski
- veber
- has_pocket_clash
- n_pocket_clashes
- vina_score
- gnina_score
- cnn_affinity
- tanimoto_ref
- generation_time_ms

## Baselines recomendados

O ideal e comparar o CODEEP contra ao menos dois tipos de baseline:

### Baseline 1: geracao 3D condicionada ao receptor

Comparadores ideais:

- liGAN
- outros metodos receptor-conditioned 3D baseados em grid, grafo ou difusao

### Baseline 2: pipeline nao generativo, mas funcional

Comparadores pragmaticos:

- ligantes conhecidos do proprio conjunto de teste
- decoys ou bibliotecas pequenas redockadas pelo mesmo rescoring
- docking tradicional sobre uma biblioteca fixa pequena

Esse segundo baseline nao responde a pergunta de geracao de novo, mas ajuda a calibrar o quao competitivo e o score funcional dos candidatos do CODEEP.

## Compatibilidade atual do CODEEP

Esta secao compara o protocolo acima com o que o CODEEP hoje ja faz e o que ainda falta.

### O que o CODEEP ja cobre

Pelo estado atual do codigo, o CODEEP ja oferece:

1. split train, val e test por cluster de PDB em vez de split aleatorio
2. metricas de treino e teste do modelo em termos de losses
3. metricas de qualidade de geracao quimica:
   - validity
   - uniqueness
   - novelty
   - internal diversity
   - QED
   - MW
   - LogP
   - SA score
   - Lipinski
4. similaridade de Tanimoto contra um ligante de referencia
5. minimizacao final da geometria do candidato
6. checagem de clashes e plausibilidade geometrica local no sitio
7. ranking interno por score composto

### O que ainda falta para compatibilidade forte com o benchmark

#### Prioridade alta

1. Integrar rescoring externo padronizado por Vina e preferencialmente gnina.
2. Salvar essas metricas em arquivo estruturado por alvo e por rodada.
3. Padronizar um runner de benchmark multialvo com seeds fixas e repeticoes.
4. Separar explicitamente metricas pre-filtro e pos-filtro.
5. Criar uma camada de avaliacao por alvo, nao apenas uma avaliacao pontual da geracao.

#### Prioridade media

1. Adicionar RMSD pos-minimizacao quando houver referencia comparavel.
2. Adicionar integracao com PoseBusters ou bateria equivalente.
3. Produzir agregados estatisticos por metodo e por alvo.
4. Criar comparadores formais com baselines externos no mesmo formato de saida.

#### Prioridade baixa

1. Teste automatizado de condicionalidade com mutacoes de residuos.
2. Benchmark de eficiencia computacional mais detalhado.
3. Dashboard consolidado de benchmark entre metodos.

## Backlog de implementacao sugerido

### Etapa A: runner de benchmark

Criar um modulo dedicado, por exemplo `MODULES/pipeline_benchmark.py`, com responsabilidades claras:

1. carregar lista de alvos e configuracao da rodada
2. chamar o gerador do CODEEP por alvo
3. aplicar filtros e minimizacao padronizados
4. disparar rescoring externo
5. agregar metricas e salvar artefatos

### Etapa B: formato padrao de resultados

Padronizar um CSV por candidato e um JSON por rodada.

Isso permite comparar CODEEP e outros metodos mesmo que a geracao venha de fora, desde que o formato final de avaliacao seja o mesmo.

### Etapa C: rescoring externo

Integrar uma interface unica para:

1. AutoDock Vina
2. gnina

O benchmark deve rodar mesmo sem gnina, mas marcar explicitamente qual bateria foi executada.

### Etapa D: agregador estatistico

Gerar automaticamente:

1. tabela por alvo
2. tabela agregada por metodo
3. ranking por metrica
4. boxplots e distribuicoes de score por alvo

## Benchmark piloto menor e mais barato

Antes do benchmark principal completo, recomenda-se um piloto com 10 a 20 alvos.

### Opcao preferencial: piloto de 10 alvos alinhado ao artigo

Usar os 10 alvos listados no artigo anexado como semente do piloto:

- 2ah9
- 5lvq
- 5g3n
- 1u0f
- 4bnw
- 4i91
- 2ati
- 2hw1
- 1bvr
- 1zyu

Vantagens:

- conjunto pequeno e barato
- ja apareceu em publicacao da area
- cobre bolsos distintos
- serve como smoke test cientificamente razoavel

### Opcao expandida: piloto de 15 a 20 alvos

Expandir os 10 alvos acima com mais 5 a 10 alvos do split de teste usando os mesmos criterios:

1. alvos de clusters distintos
2. pelo menos 5 ligantes unicos
3. diferentes tamanhos de bolso
4. diferentes familias proteicas

### Parametros do piloto

Para reduzir custo:

- 50 amostras por alvo, em vez de 100
- 2 seeds por alvo, em vez de 3
- rescoring por Vina como minimo viavel
- gnina apenas nos top-10 ou top-20 por alvo

### Metricas minimas do piloto

1. validity
2. uniqueness
3. novelty
4. internal diversity
5. QED
6. SA score
7. fracao sem clash
8. melhor Vina por alvo
9. media do top-10 Vina por alvo
10. fracao de candidatos melhores que o ligante de referencia

### Custo estimado do piloto

O piloto deve ser pensado como uma rodada de depuracao metodologica, nao como resultado final de artigo.

Configuracao recomendada:

- 10 alvos
- 50 amostras por alvo
- 2 seeds

Total bruto:

- 1000 amostras por metodo

Se o rescoring completo for caro:

1. gerar 50 por alvo
2. aplicar filtros quimicos e geometricos
3. rescoring apenas dos validos
4. opcionalmente aplicar gnina apenas nos top candidatos pre-selecionados por Vina ou score interno

## Recomendacao pratica de execucao

### Fase 1: piloto

Objetivo:

- provar que o pipeline de benchmark e executavel
- medir custo real
- identificar gargalos de rescoring e armazenamento
- validar o formato dos resultados

Configuracao:

- 10 alvos do artigo
- 50 amostras por alvo
- 2 seeds
- Vina obrigatorio
- gnina opcional nos top-10

### Fase 2: benchmark intermediario

Objetivo:

- ampliar para 15 a 20 alvos
- estabilizar estatistica por alvo
- comparar pelo menos dois metodos alem do CODEEP

Configuracao:

- 15 a 20 alvos
- 100 amostras por alvo
- 3 seeds
- Vina e gnina quando disponiveis

### Fase 3: benchmark completo

Objetivo:

- gerar tabela comparativa final para relatorio, defesa ou artigo

Configuracao:

- subconjunto robusto do split de teste CrossDocked2020
- cobertura ampla de clusters
- metricas quimicas, geometricas e funcionais completas

## Conclusao

Para comparacao seria e reprodutivel do CODEEP com outras plataformas, a melhor rota e:

1. benchmark principal sobre CrossDocked2020
2. avaliacao externa padronizada por Vina e, idealmente, gnina
3. metricas quimicas estilo MOSES ou GuacaMol como camada auxiliar
4. piloto inicial com 10 alvos do artigo anexado

No estado atual, o CODEEP ja cobre boa parte da avaliacao quimica e da plausibilidade local, mas ainda precisa principalmente de uma camada formal de benchmark multialvo e rescoring externo padronizado para ficar realmente comparavel aos trabalhos de referencia.