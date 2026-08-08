# Metodologia — CODEEP: Geração *de novo* de Ligantes Condicionada à Estrutura do Sítio de Ligação

> Rascunho da seção de Metodologia. Escrito a partir da arquitetura efetivamente implementada em
> `MODULES/pipeline_voxelize.py`, `MODULES/pipeline_train.py` e `MODULES/pipeline_generate.py`
> (commits desta sessão). Citações verificadas via busca bibliográfica onde indicado; as demais
> (clássicos de físico-química/quimioinformática) seguem a atribuição padrão da literatura — checar
> antes da submissão. Referências completas ao final.

---

## 1. Visão geral

O CODEEP formula a geração *de novo* de ligantes como um problema de aprendizado condicional:
dado o sítio de ligação de uma proteína-alvo, representado como um tensor voxelizado, um
autoencoder variacional condicional (CVAE) aprende a mapear a geometria e as propriedades
eletrônicas do bolso para um espaço latente contínuo, do qual um decoder baseado em redes neurais
de grafos (GNN) **constrói** — átomo por átomo, ligação por ligação — um ligante candidato.

O pipeline tem cinco estágios:

1. **Curadoria** — extração de pares proteína-ligante e do sítio de ligação a partir de estruturas
   cristalográficas (PDB).
2. **Protonação dependente de pH** do sítio (Seção 3).
3. **Voxelização** do sítio em um tensor 4D de 5 canais (Seção 4).
4. **Treinamento** do CVAE condicional com decoder autorregressivo (Seções 5–7), em até três fases
   de especialização progressiva (fine-tuning; Seção 8).
5. **Geração** de candidatos por amostragem sequencial em lote (*batched*) a partir do espaço
   latente, seguida de reconstrução e validação química com RDKit (Seção 9).

Um princípio de design perpassa todas as decisões arquiteturais descritas abaixo e é retomado
explicitamente na Seção 10: **paridade estrita entre a distribuição de dados vista no treino e a
disponível na inferência**. Cada canal de entrada, cada campo de features de um átomo, é
verificado quanto à sua disponibilidade tanto no momento em que o par sítio-ligante é processado
(treino) quanto no momento em que apenas a estrutura da proteína-alvo é conhecida (geração real,
sem ligante de referência).

---

## 2. Curadoria e extração do sítio de ligação

Para cada par proteína-ligante, o sítio de ligação é definido como o conjunto de resíduos
proteicos com pelo menos um átomo pesado a ≤ 6,0 Å de qualquer átomo do ligante co-cristalizado
(distância euclidiana, busca espacial via *k-d tree*). A seleção opera no nível de **resíduo
completo**: uma vez que um resíduo tem qualquer átomo dentro do raio de corte, todos os seus
átomos (cadeia principal e lateral) são incluídos — nunca um subconjunto parcial. Essa escolha é
necessária para a etapa de protonação (Seção 3), que assume fragmentos residuais completos.

---

## 3. Protonação do sítio dependente de pH

Diferente de abordagens que descartam hidrogênios do sítio (usando apenas átomos pesados com carga
aproximada por tabela fixa de elemento), o CODEEP reconstrói o estado de protonação de cada
resíduo padrão do bolso de ligação em função de um pH definido pelo usuário (padrão fisiológico,
7,4).

**Regra de decisão.** Para os sete grupos ionizáveis padrão (Asp, Glu, Cys, Tyr — ácidos; His —
tautomérico; Lys, Arg — básicos), o estado de protonação é decidido por um limiar determinístico
de Henderson–Hasselbalch sobre o pKa de tabela (Nelson & Cox — *Lehninger Principles of
Biochemistry*): protonado se pH < pKa. His é tratada à parte, com três estados possíveis (HID/HIE
neutros por tautômero, HIP⁺ duplamente protonado abaixo do pKa ≈ 6,0).

**Reconstrução geométrica.** Cada resíduo é isolado, convertido em um objeto RDKit
(`Chem.MolFromPDBBlock`, ligações inferidas por geometria), a carga formal do átomo ionizável é
ajustada conforme a decisão acima, e hidrogênios explícitos são adicionados com coordenadas 3D via
`Chem.AddHs(addCoords=True)`. As ligações peptídicas cortadas nas bordas do fragmento isolado
(backbone N/C, que na cadeia real se ligam ao resíduo vizinho) ficam com valência incompleta ao
isolar o resíduo — o RDKit completa automaticamente com H, produzindo o *capping* padrão de
métodos de química quântica fragmentada (ver Seção 4.2 e Kitaura et al., 1999). Cargas parciais
(Gasteiger–Marsili) são recalculadas para cada átomo do fragmento protonado, substituindo a tabela
aproximada por elemento usada em versões anteriores do pipeline.

A protonação é sempre re-derivada da regra de pH, ignorando hidrogênios eventualmente presentes no
PDB de origem — garante determinismo e reprodutibilidade independente da resolução/completude da
estrutura cristalográfica de entrada.

---

## 4. Voxelização: tensor do sítio em 5 canais

O sítio protonado (Seção 3) é projetado sobre uma grade cúbica regular de $N^3$ voxels (padrão
$N=32$, resolução adaptativa: o tamanho do voxel é definido de modo que a grade cubra a extensão
do sítio mais uma margem de 3,0 Å), gerando um tensor $(N, N, N, 5)$.

### 4.1 Canais 0–2 — geometria e eletrostática clássica

- **Canal 0 — densidade atômica**: soma de gaussianas 3D centradas em cada átomo, com desvio
  padrão σ = 0,5 Å (ou o tamanho do voxel, o que for maior) e amplitude dada pelo raio de van der
  Waals do elemento (Bondi, 1964).
- **Canal 1 — carga parcial**: mesma soma gaussiana, ponderada pela carga de Gasteiger–Marsili
  (Gasteiger & Marsili, 1980) de cada átomo.
- **Canal 2 — potencial eletrostático molecular (MEP)**: soma de potenciais de Coulomb pontuais,
  $\sum_i q_i / \lVert \mathbf{r} - \mathbf{r}_i \rVert$, sobre os átomos do sítio.

Os três canais são computados por um único núcleo vetorizado (`_voxelize_atoms`), com aceleração
opcional em GPU via CuPy.

### 4.2 Canais 3–4 — campo doador/aceptor de elétrons do sítio

Estes dois canais estimam o caráter nucleofílico (doador, HOMO-*like*) e eletrofílico (aceptor,
LUMO-*like*) de cada região do bolso — informação análoga aos mapas de complementaridade
doador/aceptor de métodos clássicos de campo de força (GRID; Goodford, 1985) e às *featurizations*
voxelizadas de sítios de ligação usadas em *scoring* estrutural por CNN (Ragoza et al., 2017).

**Decisão de design central — por que a partir do sítio, nunca do ligante.** Em uma versão
anterior do pipeline, estes canais eram calculados via HOMO/LUMO *ab initio* (RHF/STO-3G) do
**ligante co-cristalizado**. Isso constitui vazamento de dados treino→inferência: o encoder recebia
como parte da condição de entrada a densidade eletrônica da própria molécula que deveria aprender
a gerar — informação estruturalmente indisponível no cenário real de geração, onde não existe
ligante de referência. Como consequência, o tensor de treino e o tensor de inferência tinham
distribuições sistematicamente diferentes nesses dois canais (sempre preenchidos no treino, sempre
nulos na inferência). A correção adotada aqui — computar os canais 3–4 exclusivamente a partir dos
átomos do sítio, disponíveis em qualquer estrutura de proteína com ou sem ligante — elimina esse
vazamento por construção: o tensor é idêntico em distribuição nas duas fases.

Três níveis de teoria estão disponíveis (`site_qm_method`), do mais rigoroso ao mais rápido:

1. **RHF/STO-3G fragmentado, por resíduo** (padrão). Em vez de tentar um cálculo *ab initio* sobre
   o sítio inteiro (computacionalmente inviável — centenas de átomos), cada resíduo protonado
   (Seção 3) é tratado como um fragmento independente, no espírito do método de orbitais
   moleculares fragmentados (*Fragment Molecular Orbital*, FMO; Kitaura et al., 1999): o
   *capping* por hidrogênio nas bordas cortadas (Seção 3) produz fragmentos quimicamente bem
   formados, sempre pequenos o bastante (< 60 átomos) para caber no mesmo limite de tamanho usado
   para o ligante. O HOMO e o LUMO de cada fragmento são condensados em pesos **por átomo** via
   população de Mulliken ($c \cdot S c$, com $S$ a matriz de sobreposição de funções de base),
   evitando a geração e o realinhamento de múltiplos cubos volumétricos locais. Os pesos
   condensados são então espalhados na grade pelo mesmo núcleo gaussiano da Seção 4.1.
2. **GFN2-xTB no sítio protonado inteiro** (Bannwarth, Ehlert & Grimme, 2019), sem fragmentação —
   índices de Fukui condensados por átomo (Parr & Yang, 1984) como proxy de caráter doador (f⁻) e
   aceptor (f⁺). Escala bem para o sítio completo por ser um método semi-empírico *tight-binding*,
   mas depende de um binário externo (`xtb`) não incluído por padrão.
3. **Heurística semi-empírica** (eletronegatividade de Pauling + carga parcial) — nível mais
   rápido, sem cálculo quântico; também o *fallback* automático dos dois anteriores quando o
   cálculo real falha ou está indisponível para um resíduo específico.

Em todos os três casos, o resultado final tem a mesma forma — um peso doador e um peso aceptor por
átomo do sítio, espalhados na grade — unificando os níveis de teoria no mesmo formato de saída.

---

## 5. Arquitetura do modelo — CVAE condicional

O modelo é um autoencoder variacional condicional (Sohn, Lee & Yan, 2015 — extensão condicional do
VAE de Kingma & Welling, 2014): o encoder mapeia o tensor do sítio para os parâmetros de uma
distribuição gaussiana no espaço latente; o decoder reconstrói o grafo do ligante condicionado a
uma amostra desse espaço.

### 5.1 Encoder do sítio (`SiteEncoder`)

CNN 3D com blocos residuais (`ResBlock3D`, normalização em lote, GELU, *dropout*), três estágios de
convolução + *max-pooling* ($32^3 \to 16^3 \to 8^3 \to 4^3$), seguidos de uma MLP que produz
$(\mu, \log\sigma^2)$ de dimensão latente configurável (padrão 128–256). O tamanho do voxel é
concatenado ao vetor achatado antes da MLP, permitindo que o encoder calibre a escala física do
sítio.

### 5.2 Espaço latente

Reparametrização padrão, $z = \mu + \varepsilon \odot \sigma,\ \varepsilon \sim \mathcal{N}(0, I)$,
com $\log\sigma^2$ recortado em $[-10, 10]$ por estabilidade numérica.

### 5.3 Decoder autorregressivo (`GNNDecoder`)

Este é o componente que distingue o CODEEP de uma abordagem de *retype* sobre topologia fixa (ver
Seção 10 sobre a versão anterior) e o aproxima da família de geradores moleculares
verdadeiramente *de novo*.

**Ordem de construção — ancorada no sítio.** Para o treino por *teacher forcing*, cada ligante
precisa de uma ordem canônica de colocação de átomos. Em vez de uma ordem canônica arbitrária
(p.ex. numeração RDKit), o CODEEP ancora essa ordem na geometria do encaixe, no espírito do
Pocket2Mol (Peng et al., 2022): a construção começa no átomo do ligante real mais próximo do voxel
de maior sinal doador+aceptor do sítio (canais 3–4, Seção 4.2) e cresce por busca melhor-primeiro
sobre as ligações reais — a cada passo, expande o átomo não-visitado mais próximo espacialmente da
âncora entre os vizinhos já alcançados. Ligantes multi-fragmento (grafo desconexo) são tratados
reiniciando a busca, a cada novo fragmento, no átomo restante mais próximo da âncora. Esta ordem é
pré-computada uma vez por par e armazenada junto ao grafo (`atom_order`).

**Construção passo a passo.** No passo $t$ (com $t$ átomos já colocados), o decoder:

1. Executa uma pilha de camadas de atenção sobre grafos (GAT; Veličković et al., 2018) sobre o
   grafo **parcial** — os $t$ átomos já decididos e as ligações reais entre eles — produzindo
   embeddings por nó $h_1, \dots, h_t$ (grafo vazio se $t=0$: neste caso o contexto é só $z$).
2. Agrega um contexto por molécula, $\mathrm{ctx} = [\,\mathrm{pool}(h)\,;\, \mathrm{proj}(z)\,]$.
3. Prediz, a partir de $\mathrm{ctx}$: (a) probabilidade de **parar** (encerrar a molécula com $t$
   átomos); (b) tipo do **próximo** átomo; (c) sua carga parcial; (d) sua posição 3D.
4. Se a decisão for continuar, prediz a **ligação** do novo átomo para **cada um** dos $t$ átomos
   já existentes — tipo de ligação e probabilidade de existência — permitindo tanto ligações
   lineares de crescimento quanto fechamento de anéis (o novo átomo pode se ligar a qualquer átomo
   já colocado, não apenas ao que o "trouxe" à fronteira de busca).

**Amostragem negativa real.** Ao contrário de uma arquitetura de topologia fixa — em que todo par
testado é, por construção, uma aresta real do grafo de referência (rótulo sempre positivo) — aqui
**todo** par (novo átomo, átomo existente) entra na função de perda de existência de ligação, com
a maioria dos pares sendo negativos (não-ligados). Isso é o que efetivamente ensina o modelo a
**rejeitar** ligações, não apenas confirmá-las.

**Features de átomos já colocados — outra correção de paridade treino/inferência.** Durante o
*teacher forcing*, os átomos já posicionados poderiam, em princípio, usar suas features RDKit reais
completas (hibridização, aromaticidade, participação em anel, quiralidade) — mas essas propriedades
dependem da molécula **completa** e finalizada, informação indisponível durante a geração real
(onde a molécula ainda está sendo construída). Por isso, o vetor de features de um átomo já
colocado usa, nos dois cenários (treino e geração), apenas os campos igualmente conhecíveis nos
dois lados: tipo do átomo, número atômico, carga (predita no passo em que o átomo foi colocado) e
grau — recalculado a cada passo a partir do grafo **parcial**, não o grau da molécula final. Os
demais campos ficam zerados nos dois lados. A única assimetria residual aceita é o valor da carga
em si: no treino, o valor real (*teacher forcing* padrão); na geração, o valor que o **próprio
modelo** previu quando o átomo foi colocado — a forma clássica de *exposure bias* de modelos
autorregressivos (Bengio et al., 2015), não um vazamento binário, e com mitigação conhecida
(*scheduled sampling*) deixada como refinamento futuro.

---

## 6. Função de perda

A perda de reconstrução é a soma de seis termos, todos computados sobre os itens efetivamente
ativos em cada passo de construção (sem necessidade de máscara/*padding* adicional, pois os
tensores de predição já vêm "achatados" apenas com as posições relevantes):

$$
\mathcal{L}_{recon} = w_{stop}\,\mathcal{L}_{stop} + \mathcal{L}_{atom} + \mathcal{L}_{bond} +
\mathcal{L}_{exist} + w_{pos}\,\mathcal{L}_{pos} + 0{,}5\,\mathcal{L}_{charge}
$$

- $\mathcal{L}_{stop}$ — entropia cruzada binária da decisão parar/continuar a cada passo.
- $\mathcal{L}_{atom}$ — entropia cruzada categórica do tipo do próximo átomo (13 classes).
- $\mathcal{L}_{charge}$ — erro quadrático médio da carga parcial do próximo átomo.
- $\mathcal{L}_{pos}$ — erro quadrático médio da posição 3D (normalizada pelo desvio padrão do
  lote).
- $\mathcal{L}_{exist}$ — entropia cruzada binária da existência de cada ligação candidata,
  **com amostragem negativa real** (Seção 5.3).
- $\mathcal{L}_{bond}$ — entropia cruzada categórica do tipo de ligação (4 classes: simples, dupla,
  tripla, aromática), computada **apenas** sobre os pares com ligação real (o tipo não tem sentido
  para um par não-ligado).

A perda total inclui a divergência KL padrão do VAE, com recozimento (*annealing*) do peso β ao
longo das primeiras épocas para mitigar colapso posterior:

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + \beta \cdot D_{KL}\!\left(q(z\mid x) \,\|\, \mathcal{N}(0, I)\right), \qquad
D_{KL} = -\tfrac{1}{2}\sum_j \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

---

## 7. Paralelismo — treino e geração em lote

Um ponto frequentemente mal compreendido em geradores autorregressivos: a dependência sequencial
existe **dentro** de uma molécula (o passo $t{+}1$ depende do grafo parcial do passo $t$), não
**entre** moléculas diferentes de um mesmo lote.

- **Treino.** Com *teacher forcing*, a sequência-alvo de cada molécula já é conhecida a priori.
  O passo $t$ de **todas** as moléculas do lote é computado numa única passada batched do decoder;
  o custo real é da ordem de $\max(\text{átomos})$ passadas pequenas (≈ 60, mesmo teto usado para
  filtrar o dataset), não $B \times \max(\text{átomos})$ passadas sequenciais.
- **Geração.** Os $N$ candidatos solicitados avançam em *lockstep*: no passo $t$, uma única
  passada cobre todos os candidatos ainda ativos; candidatos que já decidiram parar saem da máscara
  ativa e não custam mais nada nos passos seguintes. É a mesma estratégia usada por decodificação
  em lote de modelos de linguagem autorregressivos.

---

## 8. Treinamento em três fases (especialização progressiva)

O treinamento segue um esquema de *fine-tuning* em cascata, típico de adaptação de domínio em
aprendizado profundo:

1. **Fase 1** — treino do zero sobre o conjunto geral (todos os pares proteína-ligante
   disponíveis), estabelecendo um prior amplo sobre química de encaixe proteína-ligante.
2. **Fase 2** — *fine-tuning* com taxa de aprendizado reduzida sobre a família/classe do alvo de
   interesse do usuário.
3. **Fase 3** — especialização final sobre alvos específicos, opcionalmente com o `SiteEncoder`
   congelado (`--freeze-encoder`), refinando apenas o decoder.

Cada fase retoma os pesos da fase anterior (`--resume`), sem herdar o estado do otimizador nem o
histórico de época — a taxa de aprendizado de cada fase é definida manualmente pelo usuário,
tipicamente decrescente a cada fase.

---

## 9. Geração e reconstrução química

Na inferência, o `SiteEncoder` calcula $\mu$ para o sítio-alvo (sem ligante de referência — Seção
4.2); $N$ vetores latentes são amostrados como $z = \mu + T \cdot \varepsilon$ (temperatura $T$
controlando diversidade), e o decoder constrói cada candidato pelo processo sequencial da Seção
5.3/7. O grafo resultante (tipos, cargas, posições, ligações já decididas pelo próprio modelo,
sem qualquer estrutura template externa) é convertido em objeto RDKit com validação química:

- checagem de compatibilidade distância↔tipo de ligação por par de elementos;
- orçamento de valência por átomo (rejeita ligações que excederiam a valência máxima do elemento);
- reparo de fragmentos desconexos residuais (liga o par válido mais próximo entre componentes,
  se necessário);
- sanitização RDKit, com *fallback* de desaromatização se a sanitização direta falhar.

**Minimização conjunta ligante+sítio e checagem de colisão estérica.** Diferente de minimizar o
ligante isoladamente e só então compará-lo a um bolso rígido (o que ou sinaliza colisões que uma
pequena reacomodação resolveria, ou aprova poses que nunca "sentiram" a proteína durante o ajuste
de geometria), a etapa final de otimização combina o candidato gerado com os átomos do sítio
(protonados, Seção 3) num único sistema — o ligante minimiza sentindo a repulsão/atração real da
proteína, não um bolso hipotético nunca considerado durante o ajuste de geometria. Dois eixos são
selecionáveis independentemente:

**Nível de teoria** (parâmetro `final_ff`), do mais simples ao mais rigoroso:

1. **MMFF94s/UFF (RDKit)** — padrão, sem dependência externa. Tenta MMFF94s primeiro; recai para
   UFF quando o sítio contém átomos que o MMFF não tipa bem (metais de cofator — Zn, Fe, Mg — o
   MMFF foi parametrizado majoritariamente para moléculas orgânicas *drug-like*, não ambientes
   proteicos metálicos). Validado empiricamente: constrói e minimiza mesmo com um átomo de Zn²⁺
   não reconhecido pelo tipador UFF.
2. **AMBER ff14SB** / **CHARMM36** (proteína) + ligante parametrizado como GAFF2 (cargas
   Gasteiger por padrão; AM1-BCC via `sqm` opcional, ~100× mais lento e sem ganho relevante aqui,
   já que o rigor adicional deste caminho vem do campo de força da *proteína*, não da carga do
   ligante), orquestrados via GROMACS (`pdb2gmx`/`editconf`/`grompp`/`mdrun`) + AmberTools
   (`antechamber`/`parmchk2`/`tleap`, chamados pelo `acpype`) — tudo por `subprocess`, nunca
   importado dentro do processo Python do CODEEP (decisão deliberada: GROMACS já é uma dependência
   do CODYN, o companheiro de dinâmica molecular do mesmo conjunto de ferramentas, então reusar seu
   binário evita uma segunda cadeia de dependências científicas Python — `openmm`+
   `openmmforcefields`+`openff-toolkit` foi cogitado e descartado por este último não ser
   instalável via `pip` em nenhuma versão do Python, só via `conda-forge`, o que forçaria um
   ambiente híbrido pip/conda só para esta funcionalidade). A topologia do sítio
   (`pdb2gmx`) é construída **uma única vez por rodada de geração** e reaproveitada por todos os N
   candidatos — o sítio não muda entre eles, só o ligante — evitando N chamadas redundantes ao
   `pdb2gmx`; cada candidato paga apenas o custo (menor) de reparametrizar seu próprio ligante via
   `acpype`. Usa os campos de força AMBER/CHARMM já empacotados com a instalação do GROMACS
   (`amber99sb-ildn`/`charmm27`) por padrão, com aviso no log; versões mais recentes
   (`amber14sb.ff`/`charmm36.ff`, baixadas separadamente) podem ser adicionadas por quem instala o
   CODEEP colocando a pasta `<nome>.ff` em `MODULES/forcefields/` — nunca no diretório
   compartilhado da instalação do GROMACS, para não afetar outras ferramentas do mesmo conjunto
   (CODYN incluso). Cai automaticamente para o caminho RDKit, com aviso no log, quando GROMACS
   (`gmx`) ou `acpype` não estão no `PATH`.

**Flexibilidade do sítio** (parâmetro `induced_fit`, desligado por padrão):

- **Rígido** (padrão): todos os átomos do sítio travados — só o ligante se move.
- **Induced-fit**: só o **backbone** (N, Cα, C, O) fica travado — representa o fato de que apenas
  o bolso recortado está presente no sistema, não a proteína inteira, então nada mais seguraria
  essa porção da cadeia no lugar. Os átomos de **cadeia lateral** ficam livres para se mover na
  mesma minimização, respondendo à presença do ligante (não é uma dinâmica térmica separada —
  é o mesmo passo de minimização de energia, com mais graus de liberdade). Isso exige que as
  ligações intra-resíduo reais estejam presentes no sistema combinado (preservadas dos fragmentos
  protonados por resíduo, Seção 3) — sem elas, nada impediria a cadeia lateral livre de colapsar
  sobre si mesma durante a otimização. Validado empiricamente: a energia total do sistema
  pós-minimização é sistematicamente menor em modo induced-fit do que em modo rígido (mais graus
  de liberdade → mínimo de energia pelo menos tão bom), embora a distância a um único par
  específico de átomos não seja necessariamente menor — a minimização otimiza a energia do
  sistema como um todo, não uma distância isolada.

Ambos os eixos combinam livremente (MMFF/UFF ou AMBER/CHARMM36, cada um rígido ou induced-fit).

Após a minimização, uma checagem de plausibilidade geométrica pós-*hoc*, no espírito do PoseBusters
(Buttenschoen, Morris & Deane, 2024), verifica: (i) colisão estérica — algum par
(átomo do ligante, átomo do sítio) mais próximo que 0,75× a soma dos raios de van der Waals
(Bondi, 1964)?; (ii) comprimento de ligação — alguma ligação do candidato fora de
[0,75, 1,25]× o comprimento canônico do seu tipo? O resultado (`has_pocket_clash`,
`n_pocket_clashes`) é anexado ao candidato e usado como fator de penalização no *score* composto
de ranqueamento (Seção 9, adiante) — não é uma checagem exaustiva (não cobre planaridade de anel,
ângulos de ligação, nem energia relativa a conformações de referência; ver limitação remanescente
na Seção 10).

Candidatos são então ranqueados por métricas de *drug-likeness* padrão (QED — Bickerton et al.,
2012; regra dos cinco de Lipinski — Lipinski et al., 2001; peso molecular; LogP), pela ausência de
colisão estérica com o sítio (acima) e, quando um ligante de referência está disponível, por
similaridade de Tanimoto sobre *fingerprints* de Morgan/ECFP.

---

## 10. Limitações metodológicas assumidas (para a Discussão)

Registradas aqui para referência direta na seção de Discussão do artigo:

1. **Ordem de construção única e determinística por ligante.** O CODEEP usa uma única ordem
   ancorada no sítio por molécula (Seção 5.3), computada uma vez e cacheada. Trabalhos como
   Pocket2Mol e MoLeR (Maziarz et al., 2022) usam esquemas de seleção de foco parcialmente
   aleatorizados durante o treino, aumentando a robustez a diferentes ordens de construção — não
   implementado aqui por simplicidade; refinamento futuro natural.
2. **Exposure bias** na carga do átomo já colocado (Seção 5.3) — mitigável por *scheduled
   sampling* (Bengio et al., 2015), não implementado nesta versão.
3. **pKa de tabela sem correção estrutural.** A protonação (Seção 3) usa pKa "de livro-texto"
   (sem ajuste por microambiente, ao contrário de ferramentas como PROPKA) — aproximação de
   primeira ordem.
4. **Nível de teoria variável nos canais 3–4** conforme o método escolhido (Seção 4.2) — RHF/STO-3G
   fragmentado, GFN2-xTB ou heurística produzem tensores com estatísticas diferentes; resultados
   de diferentes fases treinadas com métodos distintos não são diretamente comparáveis sem
   controle desse fator.
5. **Comparação com trabalho relacionado.** Ao contrário de abordagens baseadas em difusão sobre
   nuvens de pontos 3D (p.ex. DiffSBDD e sucessores diretos do Pocket2Mol), o CODEEP mantém um
   arcabouço CVAE + GNN autorregressivo — mais leve computacionalmente, mas potencialmente menos
   expressivo que modelos de difusão de última geração; uma comparação quantitativa direta
   (validade, unicidade, novidade, afinidade de *docking*) é deixada para a seção de Resultados.
6. **Plausibilidade geométrica da pose — parcialmente endereçada; sinal de afinidade ortogonal —
   ainda ausente.** Duas lacunas distintas, frequentemente confundidas sob o rótulo genérico
   "validação estrutural" — vale separá-las com precisão:
   - A checagem de colisão estérica ligante↔sítio (Seção 9) cobre o par de critérios mais
     diretamente ligados a "esse candidato colide com a proteína": distância mínima
     (vs. soma de raios de van der Waals) e comprimento de ligação pós-minimização, no espírito
     do PoseBusters (Buttenschoen, Morris & Deane, 2024). **Não** cobre o conjunto completo de
     checagens do PoseBusters — faltam ângulos de ligação, planaridade de anéis e energia da
     pose relativa a conformações de referência. A minimização em si também é uma aproximação:
     campo de força clássico (RDKit MMFF/UFF por padrão, sem parametrização específica para
     ambiente proteico), e mesmo no modo induced-fit (Seção 9) só a cadeia lateral dos resíduos
     do bolso recortado se move — sem backbone flexível, sem repacking de rotâmeros discreto, sem
     movimento de outras partes da proteína fora do bolso, e sem busca de pose completa nem
     dinâmica molecular real (é minimização de energia local a partir de uma única geometria
     inicial, não uma amostragem conformacional). Os caminhos AMBER ff14SB/CHARMM36 (via
     GROMACS+AmberTools, Seção 9) oferecem um campo de força de proteína mais rigoroso e suportam
     o mesmo esquema induced-fit — validados ponta a ponta neste ambiente de desenvolvimento
     (topologia do sítio via `pdb2gmx`, ligante via `acpype`/GAFF2, minimização com grupo de
     índice `[Frozen]` seletivo por GROMACS), incluindo a checagem de que o backbone permanece
     fixo (deslocamento residual ~10⁻⁵ Å, consistente com precisão numérica) enquanto a cadeia
     lateral se move de fato (dezenas de centésimos de Å, num teste sintético) — mas usando os
     campos de força AMBER/CHARMM já empacotados com esta instalação específica do GROMACS
     (`amber99sb-ildn`/`charmm27`, não as versões mais recentes `ff14SB`/`36`, que exigiriam baixar
     à parte a pasta `<nome>.ff` correspondente); ainda não avaliados em lote sobre um conjunto
     real de candidatos gerados pelo modelo treinado (só com ligantes sintéticos de 3 átomos, para
     isolar a corretude do encadeamento GROMACS/acpype da qualidade química do candidato em si).
   - Continua sem nenhum sinal **externo e independente** de afinidade prevista (docking com
     AutoDock Vina/Glide, ou métodos mais rigorosos como MM-GBSA/FEP) para ranquear/filtrar
     candidatos. Vale frisar o que esse tipo de sinal, se adicionado, de fato forneceria: um
     escore de docking **não é uma prova de correção estrutural** — é uma função empírica
     calibrada para correlacionar com afinidade relativa entre químicas diversas, usada para
     *priorizar* candidatos, não para *validar* que a geração está correta. Ainda assim, é
     evidência complementar padrão na literatura de geração estrutural *de novo* (Pocket2Mol,
     DiffSBDD e TargetDiff reportam o escore de Vina dos candidatos gerados como métrica
     comparativa) e vale incorporar como filtro adicional na etapa de ranqueamento
     (`rank_candidates`) em trabalho futuro.
   - RMSD contra uma pose de referência — outra métrica às vezes citada como "validação
     estrutural" — não se aplica diretamente aqui: pressupõe uma pose experimental conhecida
     para comparação átomo-a-átomo, o que só faz sentido em tarefas de *predição* de pose
     (redocking) para uma molécula de identidade fixa, não para geração *de novo*, onde o
     candidato gerado tem topologia e contagem de átomos diferentes do ligante real.

---

## Referências

- Bannwarth, C.; Ehlert, S.; Grimme, S. GFN2-xTB — An Accurate and Broadly Parametrized
  Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and
  Density-Dependent Dispersion Contributions. *J. Chem. Theory Comput.* **2019**, *15* (3),
  1652–1671.
- Bengio, S.; Vinyals, O.; Jaitly, N.; Shazeer, N. Scheduled Sampling for Sequence Prediction with
  Recurrent Neural Networks. *NeurIPS* **2015**.
- Bickerton, G. R.; Paolini, G. V.; Besnard, J.; Muresan, S.; Hopkins, A. L. Quantifying the
  Chemical Beauty of Drugs. *Nat. Chem.* **2012**, *4*, 90–98.
- Bondi, A. van der Waals Volumes and Radii. *J. Phys. Chem.* **1964**, *68* (3), 441–451.
- Buttenschoen, M.; Morris, G. M.; Deane, C. M. PoseBusters: AI-Based Docking Methods Fail to
  Generate Physically Valid Poses or Generalise to Novel Sequences. *Chem. Sci.* **2024**, *15*
  (9), 3130–3139.
- Fey, M.; Lenssen, J. E. Fast Graph Representation Learning with PyTorch Geometric. *ICLR
  Workshop on Representation Learning on Graphs and Manifolds* **2019**.
- Gasteiger, J.; Marsili, M. Iterative Partial Equalization of Orbital Electronegativity — A Rapid
  Access to Atomic Charges. *Tetrahedron* **1980**, *36* (22), 3219–3228.
- Goodford, P. J. A Computational Procedure for Determining Energetically Favorable Binding Sites
  on Biologically Important Macromolecules. *J. Med. Chem.* **1985**, *28* (7), 849–857.
- Kingma, D. P.; Welling, M. Auto-Encoding Variational Bayes. *ICLR* **2014**.
- Kitaura, K.; Ikeo, E.; Asada, T.; Nakano, T.; Uebayasi, M. Fragment Molecular Orbital Method: An
  Approximate Computational Method for Large Molecules. *Chem. Phys. Lett.* **1999**, *313* (3–4),
  701–706.
- Landrum, G. RDKit: Open-Source Cheminformatics Software. https://www.rdkit.org.
- Lipinski, C. A.; Lombardo, F.; Dominy, B. W.; Feeney, P. J. Experimental and Computational
  Approaches to Estimate Solubility and Permeability in Drug Discovery and Development Settings.
  *Adv. Drug Deliv. Rev.* **2001**, *46* (1–3), 3–26.
- Maziarz, K.; Jackson-Flux, H.; Cameron, P.; Sirockin, F.; Schneider, N.; Stiefl, N.; Segler, M.;
  Brockschmidt, M. Learning to Extend Molecular Scaffolds with Structural Motifs. *ICLR* **2022**.
  arXiv:2103.03864.
- Nelson, D. L.; Cox, M. M. *Lehninger Principles of Biochemistry*, 7th ed.; W. H. Freeman, 2017.
- Parr, R. G.; Yang, W. Density Functional Approach to the Frontier-Electron Theory of Chemical
  Reactivity. *J. Am. Chem. Soc.* **1984**, *106* (14), 4049–4050.
- Peng, X.; Luo, S.; Guan, J.; Xie, Q.; Peng, J.; Ma, J. Pocket2Mol: Efficient Molecular Sampling
  Based on 3D Protein Pockets. *ICML* **2022**, PMLR 162, 17644–17655. arXiv:2205.07249.
- Ragoza, M.; Hochuli, J.; Idrobo, E.; Sunseri, J.; Koes, D. R. Protein–Ligand Scoring with
  Convolutional Neural Networks. *J. Chem. Inf. Model.* **2017**, *57* (4), 942–957.
- Shi, C.; Xu, M.; Zhu, Z.; Zhang, W.; Zhang, M.; Tang, J. GraphAF: A Flow-Based Autoregressive
  Model for Molecular Graph Generation. *ICLR* **2020**. arXiv:2001.09382.
- Simonovsky, M.; Komodakis, N. GraphVAE: Towards Generation of Small Graphs Using Variational
  Autoencoders. *ICANN* **2018**. arXiv:1802.03480.
- Sohn, K.; Lee, H.; Yan, X. Learning Structured Output Representation Using Deep Conditional
  Generative Models. *NeurIPS* **2015**.
- Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. Graph Attention
  Networks. *ICLR* **2018**. arXiv:1710.10903.

---

### Notas de rastreabilidade (não incluir no artigo — referência interna)

| Seção | Função/arquivo |
|---|---|
| §2 | `pipeline_curate.get_binding_site` |
| §3 | `pipeline_voxelize.protonate_residue`, `protonate_site_residues`, `RESIDUE_PKA` |
| §4.1 | `pipeline_voxelize._voxelize_atoms`, `VDW_RADII` |
| §4.2 | `pipeline_voxelize.channel_site_orbital_field`, `_fragment_qm_weights`, `_xtb_whole_site_weights`, `_electronegativity_donor_acceptor_weights` |
| §5.1 | `pipeline_train.SiteEncoder` |
| §5.3 | `pipeline_train.GNNDecoder.forward_teacher_forcing`, `predict_bonds`, `_make_partial_features` |
| §5.3 (ordem) | `pipeline_voxelize.compute_site_anchored_atom_order` |
| §6 | `pipeline_train.cvae_loss` |
| §7 | `pipeline_train.LigandCVAE.sample` |
| §8 | `pipeline_train.train` (`--resume`, `--freeze-encoder`) |
| §9 | `pipeline_generate.preds_to_mol`, `compute_drug_metrics`, `rank_candidates` |
