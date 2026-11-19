# -*- coding: utf-8 -*-
"""Conteúdo do Manual do Usuário do CODEEP (pt-BR)."""
import os
FIG = os.environ.get("CODEEP_FIG", "fig")      # pasta com as capturas de tela
def f(n): return os.path.join(FIG, n)


def conteudo(m):
    # ====================================================================== 1
    m.h1("1. Descrição")
    m.p("O CODEEP é uma plataforma computacional de **geração de ligantes de novo** por aprendizado "
        "profundo, condicionada ao sítio de ligação de uma proteína-alvo. A partir da estrutura "
        "tridimensional do sítio, o programa propõe moléculas candidatas, já com coordenadas 3D, que "
        "complementam a cavidade. Seu projeto cobre todo o fluxo de trabalho: coleta de complexos "
        "proteína–ligante em bases públicas, validação e curadoria das estruturas, exploração do "
        "conjunto de dados, voxelização do sítio em tensores tridimensionais, busca de hiperparâmetros, "
        "treinamento do modelo, avaliação e geração de ligantes para uma proteína nova.")
    m.p("O modelo é um **autoencoder variacional condicional (CVAE)**. O codificador é uma rede "
        "convolucional 3D que lê o tensor do sítio — 32 canais físico-químicos calculados apenas a "
        "partir dos átomos do próprio bolso — e o resume em um vetor latente (256 dimensões, por padrão). "
        "O decodificador é uma rede neural de grafos que constrói o ligante de forma autorregressiva, "
        "átomo a átomo: a cada passo decide se a molécula termina, a partir de qual átomo já colocado "
        "ela cresce, o elemento e a carga do novo átomo, sua posição 3D e as ligações que ele faz. Na "
        "geração, o vetor latente é amostrado e o decodificador constrói moléculas para o sítio fornecido.")
    m.p("A interface guia o usuário por esse processo sem exigir conhecimento de programação. Foi "
        "desenvolvida em Python 3.10 com o framework PyQt5/Qt 5.15, em abas sequenciais que correspondem às "
        "etapas do fluxo. As principais bibliotecas integradas são PyTorch, PyTorch Geometric, RDKit, "
        "BioPython, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn e Optuna, além de PySCF, "
        "CuPy (voxelização em GPU) e do programa semiempírico GFN-xTB (`xtb`), usados no cálculo dos "
        "canais eletrônicos do sítio, e do P2Rank, usado na busca de bolsos de ligação.")
    m.nota("Neste manual, **ligantes base** são os ligantes cocristalizados curados que acompanham os "
           "complexos de treino: eles servem como resposta (alvo) para ajustar o decodificador. "
           "**Ligante de referência** é o ligante que o usuário pode fornecer, opcionalmente, na "
           "ETAPA 6 para servir de molde (*scaffold*) na avaliação dos candidatos gerados. Os dois "
           "papéis não se confundem: na geração para uma proteína nova não existe ligante base.",
           "Terminologia:")
    m.p("Este manual descreve a versão 1.1.0-1. O programa é distribuído sob a licença GPL-3.0-or-later.")

    # ====================================================================== 2
    m.h1("2. Instalação")
    m.p("O CODEEP é executado em Linux. A forma de instalação é a partir do código-fonte, descrita a seguir.")
    m.h2("2.1 Instalação a partir do código-fonte")
    m.p("Clone o repositório para a pasta pessoal e execute o programa:")
    m.codigo("sudo apt-get install git python3 python3-venv   # caso ainda não estejam instalados\n"
             "git clone https://github.com/moimaian/CODEEP.git $HOME/CODEEP\n"
             "cd $HOME/CODEEP && python3 ./CODEEP.py")
    m.p("Alternativamente, baixe o arquivo compactado do repositório "
        "(`https://github.com/moimaian/CODEEP/archive/refs/heads/main.zip`), extraia o conteúdo para a pasta "
        "pessoal renomeando-o para `CODEEP` e execute `python3 ./CODEEP.py` de dentro dessa pasta.")
    m.h2("2.2 Primeira execução e ambiente virtual")
    m.p("Na primeira execução o programa cria o ambiente virtual Python em `$HOME/.venv/CODEEP`, no qual "
        "são instaladas as dependências, e grava um atalho no menu de aplicativos "
        "(`~/.local/share/applications/CODEEP.desktop`), a partir do qual ele passa a poder ser aberto. Em "
        "seguida é exibida a janela de instalação de dependências, descrita no tópico 3.")
    m.p("Para instalar pacotes manualmente pelo terminal, o ambiente pode ser ativado com:")
    m.codigo("source $HOME/.venv/CODEEP/bin/activate")

    # ====================================================================== 3
    m.h1("3. Pré-requisitos e instalação de dependências")
    m.p("A janela de instalação de dependências é aberta na primeira inicialização e pode ser reaberta a "
        "qualquer momento pelo botão **Instalar Dependências** da aba INÍCIO ou pelo menu "
        "**Ajuda > Instalar Dependências**. Ela lista cada pacote com seu estado (instalado ou ausente), um "
        "campo de versão editável e uma descrição, instala os selecionados com o `pip` no ambiente virtual "
        "do CODEEP e mostra o andamento em um log.")
    m.figura(f("dependencias.png"),
             "Janela de instalação de dependências, com o estado e a versão de cada pacote.", 14.5)
    m.p("A versão do PyTorch é escolhida de acordo com o hardware detectado: a variante para CUDA 12 "
        "(`cu121`) quando há GPU NVIDIA compatível, a de CUDA 11 (`cu118`) para instalações mais antigas e a "
        "variante apenas para CPU quando não há GPU. Os pacotes são classificados em três grupos:")
    m.tabela(["Grupo", "Pacotes"], [
        ["Essenciais", "PyQt5, PyTorch, torch-geometric, NumPy, Pandas, SciPy, Matplotlib, scikit-learn, "
                       "RDKit, BioPython"],
        ["Opcionais", "Seaborn, Optuna (necessário só para a ETAPA 3), PySCF, GPU4PySCF, CuPy e cuTENSOR "
                      "(aceleração em GPU), tqdm, psutil, requests, openpyxl"],
        ["Ferramentas externas", "`xtb` (GFN-xTB) e P2Rank 2.5.1, baixados pela própria janela para a pasta "
                                 "`tools/`; Java 17 ou superior (dependência de sistema, exigida pelo P2Rank); "
                                 "GROMACS e AmberTools, opcionais, só para os campos de força AMBER/CHARMM da ETAPA 6"],
    ], [3.4, 12.6])
    m.p("Dois itens merecem atenção:")
    m.bullets([
        "O **GFN-xTB** é o método recomendado para calcular os canais doador/aceptor do sítio na ETAPA 2. "
        "Sem o binário `xtb` o programa recorre automaticamente a uma heurística baseada em eletronegatividade, "
        "e registra no arquivo `meta.json` de cada par o método realmente usado;",
        "A **GPU** não é obrigatória, mas é praticamente necessária para o treinamento (ETAPA 4) em escala. A "
        "voxelização também é acelerada por GPU quando o CuPy está presente; sem ele, usa NumPy em CPU."])
    m.p("A aba INÍCIO mostra as especificações de CPU, GPU e armazenamento da máquina e as versões "
        "efetivamente detectadas de cada pacote e ferramenta (grupo Especificações de Software). Essa é a forma "
        "mais rápida de conferir se o ambiente está completo.")
    m.figura(f("inicio.png"),
             "Aba INÍCIO: especificações de hardware e software detectadas e resumo das etapas do pipeline.", 14.5)
    m.p("A caixa **Ajustar parâmetros ao Hardware** adapta ao equipamento detectado os campos que mais pesam no "
        "desempenho: máximo de entradas da Fase 1, precisão do tensor, método dos canais HOMO/LUMO, máximo de "
        "átomos QM, tamanho do lote de voxelização, tamanho do lote e taxa de aprendizado do treino e "
        "dispositivo. Ao marcá-la, cada alteração é mostrada com a respectiva justificativa; ao desmarcá-la, os "
        "valores anteriores são restaurados. O botão **Ver Terminal** abre um terminal integrado.")

    # ====================================================================== 4
    m.h1("4. Organização dos diretórios")
    m.p("O CODEEP trabalha dentro da pasta onde se encontra o arquivo `CODEEP.py`, que é identificada "
        "automaticamente. Os dados da Fase 1, compartilhados entre todos os projetos, ficam em `data/`; cada "
        "projeto tem sua pasta própria em `PROJECTS/`. A organização é a seguinte:")
    m.codigo("""CODEEP/
├── CODEEP.py                      arquivo principal
├── CODEEP_requirements.py         verificação e instalação de dependências
├── MODULES/                       módulos do pipeline (coleta, curadoria, voxelização, treino...)
├── ICONS/                         ícones e logotipos
├── TUTORIALS/                     manuais do usuário
├── tools/                         binários auxiliares (xtb, P2Rank)
├── data/                          dados da Fase 1, compartilhados entre os projetos
│   ├── raw/                       estruturas baixadas (pdb/, sdf/)
│   ├── curated/<par>/             receptor.pdb, ligand.pdb, binding_site.pdb
│   ├── tensors/<par>/             site_tensor.npz, ligand_graph.pt, meta.json
│   ├── figures/                   figuras de exploração e de avaliação
│   └── logs/                      logs de cada etapa e diagnósticos
├── checkpoints/phase1/            modelo treinado da Fase 1
├── hpo/                           resultados da otimização de hiperparâmetros
└── PROJECTS/
    └── <data>_<nome>/             uma pasta por projeto
        ├── job_config.json        identificação do projeto e termos de busca
        ├── step*.json             estado salvo de cada aba
        ├── data/                  dados das Fases 2 e 3 (mesma estrutura)
        ├── checkpoints/           phase2/ e phase3/
        └── results/<alvo>/        candidatos gerados (ETAPA 6)""")
    m.p("A pasta de cada par proteína–ligante recebe o nome `<código PDB>_<ligante>_<cadeia><resíduo>`, por "
        "exemplo `10gs_VWW_A210`. Os tensores ocupam cerca de 1,5 MB por par em precisão float32 e cerca de "
        "0,45 MB em float16, valores medidos no conjunto de 32 canais; o grupo Especificações de Hardware — "
        "Armazenamento da aba INÍCIO estima quantos pares cabem no disco disponível em cada precisão.")
    m.nota("O estado de todas as abas — parâmetros, caminhos e opções — é salvo automaticamente na pasta do "
           "projeto e restaurado ao reabri-lo. Os arquivos `step*.json` mantêm a numeração interna "
           "anterior das abas: `step1.json` guarda a CONFIG; `step2.json`, `step2cur.json` e `step2exp.json`, a "
           "ETAPA 1; `step3.json`, a ETAPA 2; `step4.json`, a ETAPA 3; `step5.json`, a ETAPA 4; `step6.json`, "
           "a ETAPA 5; e `step7.json`, a ETAPA 6.")

    # ====================================================================== 5
    m.h1("5. Orientações gerais de uso")
    m.p("A janela principal é organizada em abas sequenciais que correspondem à ordem natural do fluxo de "
        "trabalho: **INÍCIO**, **CONFIG** e **ETAPA 1 a ETAPA 6**. A navegação pode ser feita pelas abas, pelos "
        "botões **VOLTAR** e **AVANÇAR** no rodapé de cada aba ou pelo menu **Menu**. A barra de menus tem três "
        "entradas: **Menu** (Início, CONFIG, as seis etapas e Sair), **Ajuda** (Instalar Dependências e Gerar "
        "Relatório Final) e **Sobre**.")
    m.p("No canto superior direito ficam as bandeiras para alternar entre português e inglês — a escolha é "
        "memorizada entre execuções — e o botão de monitor de CPU/GPU, que abre uma janela com o uso de "
        "processador, memória e placa de vídeo durante as execuções mais demoradas.")
    m.p("Em praticamente todos os campos há textos de ajuda, exibidos ao posicionar o cursor sobre o "
        "elemento. Eles trazem a descrição detalhada do parâmetro, os valores recomendados e as consequências "
        "de cada escolha, e são a referência mais completa e atualizada sobre cada controle.")
    m.p("Os parâmetros de cada aba estão organizados em **grupos recolhíveis**: clicar no título de um grupo "
        "(▾ aberto, ▸ recolhido) o fecha ou o abre, o que encurta a aba quando não se quer ver todos os "
        "campos. As abas de execução (ETAPAS 1 a 6) compartilham o mesmo conjunto de controles na parte "
        "inferior:")
    m.bullets([
        "**Executar** e **Parar**: o processamento roda em um processo separado da interface. Parar encerra "
        "esse processo, sem fechar o CODEEP. Enquanto um grupo da ETAPA 1 está em execução, os botões "
        "Executar dos outros ficam desabilitados, pois os três compartilham um único log;",
        "**Log de Execução**: seção recolhível com a saída em tempo real do processamento. Ao ser aberta, a "
        "janela cresce o necessário para que o conteúdo não fique sobreposto, e volta ao tamanho anterior "
        "quando a seção é fechada;",
        "**Barra de progresso**: mostra o percentual concluído e o tempo restante estimado."])
    m.p("O fechamento da janela com uma etapa em andamento é tratado com confirmação, para evitar a "
        "interrupção acidental de um treinamento longo.")
    m.h2("5.1 As três fases de dados")
    m.p("Em várias abas há o campo **Fase**, que indica sobre qual conjunto de dados a etapa atua. A "
        "estratégia de três fases permite partir de um modelo geral e especializá-lo:")
    m.bullets([
        "**Fase 1 — pré-treino geral.** Usa um conjunto externo amplo de complexos proteína–ligante "
        "(lp-PDBbind, RCSB, PLINDER ou Binding MOAD). Seus dados são compartilhados entre todos os projetos e "
        "ficam em `data/` e `checkpoints/phase1/`;",
        "**Fase 2 — especialização por família ou organismo.** Busca no RCSB por organismo e por descrição da "
        "entidade, com os termos definidos na aba CONFIG. Os dados e o modelo ficam na pasta do projeto, em "
        "`data/.../phase2` e `checkpoints/phase2`;",
        "**Fase 3 — alvos específicos.** Baixa estruturas a partir de uma lista de códigos PDB informada na "
        "CONFIG. Os dados e o modelo ficam em `phase3`."])
    m.p("No treinamento, a Fase 2 parte, por padrão, do modelo da Fase 1 (ajuste fino) e a Fase 3 parte do "
        "modelo da Fase 2 ou, na ausência dele, do da Fase 1. Em todos os casos, o campo **Retomar de** "
        "permite escolher outro ponto de partida.")

    # ====================================================================== 6
    m.h1("6. Aba CONFIG — configuração do projeto")
    m.p("A aba CONFIG define o projeto de trabalho. É o primeiro passo: sem um projeto criado ou carregado, as "
        "etapas não têm onde gravar seus resultados.")
    m.figura(f("config.png"),
             "Aba CONFIG: identificação do projeto, caminhos detectados e termos de busca das Fases 2 e 3.", 14.5)
    m.h2("6.1 Identificação do projeto")
    m.p("O grupo **Identificação do Projeto** segue esta ordem:")
    m.numerada([
        "**Data** — somente leitura, mostra a data e a hora atuais;",
        "**Novo Projeto** — quando marcado, habilita o campo **Nome do Job**, onde se digita o nome do novo "
        "projeto;",
        "**Carregar Projeto Anterior** — quando marcado, habilita a lista **Projeto Anterior**, com os projetos "
        "existentes em `PROJECTS/`. As duas opções são mutuamente exclusivas;",
        "**ID do Job** — exibe o nome do projeto criado ou carregado depois de acionado o botão "
        "**Criar / Carregar Projeto**."])
    m.p("Ao criar um projeto novo, são criadas a estrutura global da Fase 1 (se ainda não existir) e a pasta do "
        "projeto, nomeada `<data>_<hora>_<nome>`, por exemplo `2026-11-14_00-49_TESTE_ARCH5`, com os "
        "subdiretórios das Fases 2 e 3. Ao carregar um projeto existente, o estado salvo de todas as abas é "
        "restaurado, inclusive as curvas de treinamento, quando houver histórico no disco.")
    m.h2("6.2 Caminhos detectados")
    m.p("O grupo **Caminhos Detectados** mostra o estado das pastas de dados e de checkpoints, separadas em "
        "**Global (compartilhado — Fase 1)** e **Específico do Job (Fase 2 / 3)**. Para cada pasta, o sinal "
        "✔ ou ✘ indica se ela existe, acompanhado da contagem de arquivos encontrados (PDB, SDF, `.pt`, `.npz`). É a "
        "forma rápida de saber até onde o pipeline já avançou em cada fase.")
    m.h2("6.3 Termos de busca das Fases 2 e 3")
    m.p("O grupo **Fase 2 / 3 — Termos de Busca do Alvo** reúne três áreas de texto, uma entrada por linha:")
    m.bullets([
        "**Termos de organismo** — usados na busca por taxonomia no RCSB (Fase 2), por exemplo `Homo sapiens`;",
        "**Termos de descrição da entidade** — usados na busca por palavra-chave no RCSB (Fase 2), por exemplo "
        "`protein kinase`;",
        "**Fase 3 — IDs de PDB específicos** — códigos PDB separados por vírgula ou por quebra de linha, por "
        "exemplo `1ATP`, `2SRC`."])

    # ====================================================================== 7
    m.h1("7. ETAPA 1 — Gerenciamento de dados")
    m.p("Esta etapa reúne, em uma só aba, três blocos independentes: a **Coleta de Dados**, a **Validação e "
        "Curadoria de Dados** e a **Exploração de Dados**. Cada bloco tem seus próprios parâmetros e seu botão "
        "Executar, mas o log de execução e a barra de progresso são únicos para a aba.")
    m.figura(f("etapa1.png"),
             "Aba ETAPA 1: coleta, validação/curadoria e exploração dos dados.", 14.5)
    m.h2("7.1 Coleta de dados")
    m.p("A coleta baixa as estruturas e os ligantes das bases escolhidas. Os controles do grupo são:")
    m.bullets([
        "**Dataset da Fase 1** — escolhe a base externa: `lp-pdbbind` (cerca de 17.000 pares curados, sem "
        "vazamento de dados; é a opção padrão), `rcsb-wide` (busca ampla no RCSB por complexos com resolução de "
        "até 2,5 Å), `plinder` (cerca de 449.000 pares, a maior base disponível) ou `moad` (Binding MOAD, cerca "
        "de 41.000 pares, que exige registro prévio no site da base);",
        "**Fase de Coleta** — Fase 1, Fase 2, Fase 3 ou Todas as Fases em sequência. A Fase 2 usa os termos de "
        "organismo e de descrição, e a Fase 3 usa os códigos PDB, ambos definidos na CONFIG;",
        "**Máx. de entradas da Fase 1** — limita o número de entradas baixadas (padrão 50.000);",
        "**Pasta MOAD** — obrigatória apenas para o dataset MOAD, que deve ser baixado previamente;",
        "**Mostrar apenas o status dos dados coletados** — não baixa nada, apenas relata o que já existe."])
    m.h2("7.2 Validação e curadoria")
    m.p("A curadoria lê os arquivos PDB baixados, valida a qualidade das estruturas, extrai o sítio de "
        "ligação e produz o conjunto curado usado nas etapas seguintes. Para cada par aprovado são gravados "
        "`receptor.pdb` (proteína sem ligante e sem água), `ligand.pdb` (o ligante base, na posição do cristal), "
        "`binding_site.pdb` (resíduos do sítio) e `pair_info.json`. Os parâmetros são:")
    m.tabela(["Campo", "Padrão", "Função"], [
        ["Fase", "1", "Conjunto de dados a curar."],
        ["Trabalhadores (0=auto)", "0", "Número de processos em paralelo; 0 detecta automaticamente."],
        ["Cutoff do sítio (Å)", "6,0", "Distância máxima entre um resíduo e o ligante para que o resíduo "
                                      "faça parte do sítio."],
        ["Resíduos mín. do sítio", "5", "Estruturas cujo sítio tem menos resíduos que isso são rejeitadas."],
        ["MW máx. do ligante (Da)", "800", "Ligantes mais pesados são rejeitados."],
        ["Átomos pesados mín.", "7", "Ligantes menores são rejeitados."],
        ["ThreadPoolExecutor / ProcessPoolExecutor", "Processos", "Tipo de paralelismo. Processos é o padrão, "
                                                                "pois a leitura dos PDB com o BioPython não se "
                                                                "beneficia de threads."],
    ], [4.8, 2.2, 9.0])
    m.p("Além desses filtros, a curadoria remove automaticamente solventes, íons, tampões, aditivos de "
        "cristalização, polietilenoglicóis, detergentes e lipídios, e rejeita ligantes ligados "
        "covalentemente à proteína (como resíduos modificados e glicosilações), pois não são ligantes no sentido "
        "do problema. Nucleotídeos e cofatores também são excluídos por padrão: concentram o elemento fósforo e "
        "o núcleo purina e enviesariam a composição elementar do modelo.")
    m.nota("Desligar a exclusão de cofatores é recomendado apenas quando o alvo for, justamente, uma enzima "
           "dependente de nucleotídeo. Essa opção está em `CURATE_CONFIG` no módulo `pipeline_curate.py`.")
    m.h2("7.3 Exploração de dados")
    m.p("O botão **Ver Painel** gera, para a fase escolhida, o painel de análise exploratória do conjunto "
        "curado e o exibe em uma janela. O painel reúne as distribuições do peso molecular, do tamanho dos ligantes "
        "e do tamanho dos sítios, os ligantes mais frequentes e estatísticas resumidas, e é a forma recomendada de conferir o conjunto antes da "
        "voxelização. As figuras são gravadas em `data/figures/`, em PNG e SVG.")
    m.figura(f("summary_dashboard.png"),
             "Painel de exploração do conjunto curado (Ver Painel).", 14.0)

    # ====================================================================== 8
    m.h1("8. ETAPA 2 — Voxelização")
    m.p("Esta etapa converte cada par curado em dados que a rede neural consome: um **tensor 3D do sítio** e "
        "um **grafo molecular do ligante base**. Para cada par, são gravados em `data/tensors/<par>/` os "
        "arquivos `site_tensor.npz` (tensor comprimido), `ligand_graph.pt` (grafo no formato do PyTorch "
        "Geometric, com tipos de átomo, cargas, ligações e coordenadas 3D do cristal) e `meta.json`.")
    m.figura(f("etapa2.png"),
             "Aba ETAPA 2: escopo, preparação do sítio, grade de voxels e canais quânticos.", 14.5)
    m.p("O tensor do sítio tem **32 canais**, todos calculados exclusivamente a partir dos átomos do próprio "
        "bolso de ligação e nunca do ligante. Isso garante que o tensor seja idêntico no treinamento e na "
        "geração, quando não há ligante algum. Os canais formam três grupos:")
    m.tabela(["Grupo", "Canais", "Conteúdo"], [
        ["Elemento", "5", "Densidade gaussiana separada por espécie: C, N, O, S e metais de coordenação."],
        ["Físico-química", "6", "Hidrofobicidade (Crippen), aromaticidade, carga parcial (Gasteiger), potencial "
                                "eletrostático (Coulomb) e os campos doador e aceptor de elétrons (semelhantes a "
                                "HOMO e LUMO)."],
        ["Aminoácido", "21", "Identidade do resíduo, um canal para cada um dos 20 aminoácidos canônicos e um "
                             "para os demais."],
    ], [3.2, 1.8, 11.0])
    m.h2("8.1 Escopo e execução")
    m.bullets([
        "**Fase** — conjunto a voxelizar (1, 2 ou 3);",
        "**Par específico** — processa apenas o par indicado; vazio processa todos;",
        "**Trabalhadores** e **Paralelizar pares** — por padrão os pares são processados um por vez, o "
        "comportamento validado. Marcar Paralelizar pares processa vários simultaneamente, dividindo as "
        "threads entre eles; o ganho esperado é modesto, de 15 a 25%, porque o cálculo quântico, a parte mais "
        "lenta, já usa quase todos os núcleos em cada par;",
        "**Tamanho do lote de voxel** — átomos por lote na voxelização gaussiana (padrão 500; 500 é "
        "recomendado para GPU com 8 a 16 GB e 100 para CPU). Valores maiores reduzem o custo por lote, mas "
        "aumentam o uso de memória;",
        "**Precisão do tensor** — precisão com que o tensor é gravado em disco, `float32` (padrão) ou "
        "`float16`. Não altera o cálculo nem o treino; `float16` reduz o espaço em disco a cerca de um terço, "
        "ao custo de um pequeno erro de arredondamento."])
    m.h2("8.2 Preparação do sítio")
    m.bullets([
        "**Protonar resíduos do sítio (baseado em pH)** — adiciona hidrogênios explícitos aos resíduos "
        "ionizáveis (Asp, Glu, His, Cys, Tyr, Lys e Arg) segundo o pKa de tabela e o pH escolhido, e passa a "
        "usar cargas de Gasteiger por átomo. Desligar volta ao comportamento legado, sem hidrogênios;",
        "**pH do sítio** — pH usado para decidir o estado de protonação (padrão 7,4)."])
    m.h2("8.3 Grade de voxels e canais")
    m.bullets([
        "**Tamanho da grade (N³)** — número de voxels por lado (padrão 32, ou seja, 32 × 32 × 32);",
        "**Preenchimento (Å)** — margem adicionada ao redor do sítio ao definir a caixa (padrão 3,0 Å). O "
        "tamanho do voxel resulta da maior dimensão do sítio, acrescida do preenchimento, dividida pela "
        "grade;",
        "**Usar ECFP4 no grafo** — acrescenta ao grafo do ligante a impressão digital ECFP4. Reduz a "
        "diversidade e vem desmarcada."])
    m.h2("8.4 Canais quânticos (QM)")
    m.bullets([
        "**Método dos canais HOMO/LUMO do sítio** — nível de teoria para o campo doador/aceptor: "
        "**GFN-xTB por resíduo, paralelo** (recomendado, requer o binário `xtb` e leva cerca de 0,25 s por "
        "par), **RHF/STO-3G por resíduo** (cálculo quântico fragmentado, mais lento) ou **Heurística** "
        "(eletronegatividade, sem cálculo quântico);",
        "**Máx. átomos QM** — limite de átomos pesados para o cálculo completo (padrão 60);",
        "**Base QM** — base do método RHF por resíduo (`sto-3g`, `3-21g` ou `6-31g`); é ignorada pelo "
        "GFN-xTB e pela heurística."])
    m.nota("O método e a base escolhidos são gravados no checkpoint treinado. Na ETAPA 6, a geração usa "
           "automaticamente a configuração de voxelização do treino, de modo que o tensor do sítio novo tenha "
           "a mesma distribuição do conjunto de treinamento.")

    # ====================================================================== 9
    m.h1("9. ETAPA 3 — Otimização de hiperparâmetros")
    m.p("Esta etapa é opcional. Ela usa o Optuna para buscar combinações de hiperparâmetros em um "
        "subconjunto barato dos dados e em poucas épocas por tentativa (*trial*). Como o ranking vem de um "
        "subconjunto e/ou de épocas iniciais, a busca é um **proxy**: use-a para escolher duas ou três "
        "configurações e confirme-as com um treinamento completo na ETAPA 4.")
    m.figura(f("etapa3.png"),
             "Aba ETAPA 3: orçamento da busca e espaço de busca dos hiperparâmetros.", 14.5)
    m.h2("9.1 Orçamento da busca")
    m.bullets([
        "**Algoritmo de busca** — define qual configuração testar a seguir: TPE (Optuna, padrão), Random "
        "Search, Grid Search ou Bayesiano (Processo Gaussiano). No Grid Search, *Trials* passa a ser o teto "
        "da grade;",
        "**Fase (dataset)**, **Trials** (padrão 40), **Épocas por trial** (padrão 15) e **Fração do dataset** "
        "(padrão 0,20);",
        "**ASHA — épocas mínimas** e **fator de redução** — configuram o Successive Halving, que é um algoritmo "
        "de *poda*: decide quando interromper uma tentativa em andamento e combina com qualquer algoritmo de "
        "busca;",
        "**Semente**, **Dispositivo** e **Nome do estudo** — reprodutibilidade, GPU ou CPU, e nome do estudo "
        "Optuna;",
        "**Retomar de** — `best_model.pt` de um treino anterior, para otimizar o ajuste fino nas Fases 2 e 3; "
        "**Congelar encoder** — mantém o codificador fixo nesse ajuste;",
        "**Manter checkpoints de todos os trials** — consome muito mais disco;",
        "**Pasta de saída** — padrão `hpo/`."])
    m.h2("9.2 Espaço de busca")
    m.p("Cada hiperparâmetro tem uma caixa de seleção e uma faixa: `mín,máx` para parâmetros contínuos ou "
        "inteiros e uma lista de valores para os categóricos. A faixa só é editável com a caixa marcada. Os "
        "parâmetros estão divididos em três colunas por prioridade; os de alta e média prioridade já vêm "
        "marcados:")
    m.tabela(["Prioridade", "Hiperparâmetros"], [
        ["Alta", "`lr`, `beta_max`, `pos_weight`, `stop_weight`"],
        ["Média", "`beta_warmup`, `dropout`, `hidden_dim`, `n_gnn_layers`"],
        ["Baixa", "`latent_dim`, `batch_size`, `weight_decay`, `grad_clip`"],
    ], [3.0, 13.0])
    m.p("O **Objetivo** pode ser `val_recon` (apenas a reconstrução, invariante a beta) ou `val_total` "
        "(reconstrução mais beta·KL). Como `val_total` só é válido com beta fixo, ele é desabilitado "
        "automaticamente enquanto `beta_max` ou `beta_warmup` estiverem no espaço de busca. Os botões "
        "**Marcar todos**, **Restaurar padrões** e **Desmarcar todos** agem sobre as caixas, e **Estimar espaço em "
        "disco** calcula quanto o estudo vai ocupar. Atenção: `batch_size` interage "
        "fortemente com `lr`; buscá-lo isoladamente engana, e só faz sentido com `lr` também marcado.")
    m.p("Ao final, a pasta de saída contém `best_params.json` (melhor combinação), `hpo_trials.csv` (todas as "
        "tentativas) e `optuna_study.db` (o estudo, que pode ser retomado).")

    # ====================================================================== 10
    m.h1("10. ETAPA 4 — Treinamento")
    m.p("Esta etapa treina o CVAE sobre os pares voxelizados. O codificador recebe o tensor do sítio e produz "
        "a distribuição do vetor latente; o decodificador, em *teacher forcing*, recebe o vetor latente e o "
        "grafo parcial do ligante base e aprende a prever o próximo passo da construção. O treino otimiza a "
        "perda de reconstrução e o termo de divergência de Kullback–Leibler (KL), ponderado por um beta que "
        "cresce gradualmente (*warmup*).")
    m.figura(f("etapa4.png"),
             "Aba ETAPA 4: parâmetros do treinamento, organizados em grupos recolhíveis.", 14.5)
    m.p("A divisão em treino, validação e teste (80%, 10% e 10%) é feita pelo **código PDB**: todos os "
        "ligantes de uma mesma estrutura ficam na mesma partição. Não é feita separação por similaridade de "
        "sequência entre proteínas diferentes.")
    m.h2("10.1 Parâmetros")
    m.tabela(["Grupo", "Campo", "Padrão", "Função"], [
        ["Capacidade", "Dim. latente", "256", "Dimensão do vetor latente."],
        ["", "Dim. oculta", "256", "Largura do decodificador GNN."],
        ["", "Camadas GNN", "4", "Profundidade do decodificador."],
        ["Otimização", "Épocas", "150", "Número máximo de épocas."],
        ["", "Tamanho do lote", "32", "Pares por passo."],
        ["", "Taxa de aprendizado", "4e-4", "Taxa do otimizador AdamW."],
        ["", "Paciência", "15", "Épocas sem melhora na validação antes de interromper (*early stopping*)."],
        ["Regularização", "Dropout", "0,20", "Regularização do modelo."],
        ["", "Weight decay", "1e-4", "Regularização L2 do AdamW."],
        ["", "Recorte do gradiente", "0,50", "Norma máxima do gradiente por passo; protege contra explosão do "
                                              "gradiente."],
        ["Pesos da perda e KL", "Beta KL (máx.)", "0,5", "Peso máximo do termo KL; é o principal responsável "
                                                         "pelo colapso de posterior."],
        ["", "Beta warmup", "30", "Épocas até o beta atingir o máximo."],
        ["", "Peso da posição", "2,0", "Peso do termo de posição 3D. É uma log-verossimilhança, não um erro "
                                       "em ångström."],
        ["", "Peso da parada", "1,0", "Controla o tamanho das moléculas: baixo gera ligantes que não param "
                                      "de crescer, alto gera fragmentos."],
        ["", "Peso do foco", "1,0", "Peso da escolha do átomo a partir do qual a molécula cresce."],
        ["Scheduled sampling", "Amostragem programada", "0,25", "Fração máxima de átomos do grafo parcial "
                                                               "mostrados ao decodificador com o elemento que "
                                                               "ele próprio previu; 0 desliga."],
        ["", "Início", "20", "Épocas de *teacher forcing* puro antes de a substituição começar."],
        ["", "Rampa", "40", "Épocas para ir de 0 ao máximo."],
        ["Execução", "Fase", "1", "Conjunto de treino."],
        ["", "Dispositivo", "cuda", "GPU ou CPU, a partir das GPUs detectadas."],
        ["Checkpoints", "Retomar de", "—", "`best_model.pt` de onde partir (ajuste fino)."],
        ["", "Salvar checkpoint em", "checkpoints/", "Pasta de saída."],
        ["", "Congelar encoder", "desmarcado", "Mantém o codificador fixo (ajuste fino da Fase 3)."],
    ], [3.3, 3.6, 2.2, 6.9])
    m.h2("10.2 Acompanhamento")
    m.p("A seção **Curvas de Treinamento**, recolhida ao abrir a aba, mostra em tempo real a perda de "
        "treino e de validação a cada época; ao expandi-la, a janela cresce para acomodá-la. O **Log de "
        "Execução** traz, por época, as perdas e as métricas de diagnóstico, entre elas o erro geométrico "
        "médio das posições (`posMAE`, em Å).")
    m.p("Ao final do treino, a pasta de checkpoints contém:")
    m.bullets([
        "`best_model.pt` — o modelo com a menor perda de validação, que carrega também a configuração "
        "usada no treino (arquitetura, referencial das posições e parâmetros de voxelização);",
        "`epoch_NNNN.pt` — pesos gravados a cada 10 épocas;",
        "`training_history.csv` e `training.log` — histórico por época e registro completo do treino."])
    m.nota("Uma queda do KL a valores próximos de zero indica *colapso de posterior*: o decodificador passa "
           "a ignorar o vetor latente e a geração perde diversidade. O diagnóstico de unidades ativas, na "
           "ETAPA 5, ajuda a detectar essa situação.")

    # ====================================================================== 11
    m.h1("11. ETAPA 5 — Avaliação")
    m.p("Esta etapa avalia o modelo treinado. O botão **Executar** roda, em sequência, a análise do "
        "histórico de treinamento (épocas treinadas, melhor época, lacuna entre treino e validação, KL final e "
        "problemas detectados), a avaliação no conjunto de teste, o diagnóstico do espaço latente e, "
        "opcionalmente, a geração de ligantes de exemplo.")
    m.figura(f("etapa5.png"),
             "Aba ETAPA 5: modelo e dados, e os três botões de figuras e diagnóstico.", 14.5)
    m.h2("11.1 Modelo e dados")
    m.bullets([
        "**Fase** — fase do modelo a avaliar;",
        "**Checkpoints** — pasta do modelo; vazia, usa `checkpoints/phase<N>/` da fase escolhida;",
        "**Gerar ligantes durante a avaliação** — além das métricas, gera alguns ligantes para calcular "
        "validade, unicidade e propriedades químicas."])
    m.h2("11.2 Figuras e diagnósticos")
    m.p("O grupo **Figuras e Diagnósticos** tem três botões na mesma linha, na ordem em que costumam ser "
        "usados:")
    m.bullets([
        "**Curvas de Treinamento** — exibe a figura `training_curves`, com a evolução das perdas ao longo "
        "das épocas;",
        "**Relatório de Desempenho** — exibe a figura `performance_report`, com a perda de teste por "
        "componente, a distribuição de QED dos ligantes gerados e uma tabela de métricas de geração;",
        "**Diagnóstico de Active Units** — executa o diagnóstico descrito no tópico 11.3."])
    m.p("As figuras são gravadas em PNG e SVG em `data/figures/` (Fase 1) ou na pasta de figuras do projeto "
        "(Fases 2 e 3). Elas só existem depois que a avaliação foi executada.")
    m.figura(f("training_curves.png"), "Curvas de treinamento do modelo da Fase 1.", 14.0)
    m.figura(f("performance_report.png"),
             "Relatório de desempenho: perdas no conjunto de teste, distribuição de QED e métricas de geração.", 16.0)
    m.h2("11.3 Diagnóstico de Active Units")
    m.p("Verifica quantas das dimensões do espaço latente o codificador realmente usa, pela métrica de "
        "*active units* (Burda, Grosse e Salakhutdinov, 2016). Para cada dimensão, calcula-se a variância, entre "
        "os sítios do conjunto de validação, da média posterior; a dimensão é considerada ativa quando essa "
        "variância supera 0,01. O diagnóstico detecta o colapso **parcial** de posterior, que um valor médio "
        "de KL, sozinho, pode esconder: um KL médio moderado ainda pode significar poucas dimensões ativas e "
        "o resto morto.")
    m.p("O resultado aparece em uma janela de texto, com o checkpoint, a época salva, o conjunto e o número de "
        "amostras usados, o limiar, o total de unidades ativas e uma interpretação: boa utilização (50% ou mais "
        "ativas), utilização parcial (de 10% a 50%) ou forte indício de colapso parcial (menos de 10%).")
    m.figura(f("active_units.png"),
             "Janela do Diagnóstico de Active Units, com o caminho do log gravado.", 12.0)
    m.p("Cada execução também é **gravada em disco**, na pasta de logs da fase escolhida (`data/logs/` na "
        "Fase 1 e `<projeto>/data/logs/phase<N>/` nas Fases 2 e 3), para consulta posterior e uso em relatórios:")
    m.bullets([
        "`active_units.log` — relatório em texto; cada execução é acrescentada ao final, com data e hora;",
        "`active_units.json` — a execução mais recente em formato estruturado, com checkpoint, época, "
        "conjunto, amostras, limiar, unidades ativas, total, fração e a variância de cada dimensão."])
    m.p("O botão **Abrir log** da janela abre o arquivo no aplicativo padrão do sistema.")

    # ====================================================================== 12
    m.h1("12. ETAPA 6 — Geração de candidatos a ligante")
    m.p("Esta etapa gera ligantes candidatos para uma proteína que o modelo não viu no treinamento. A partir "
        "do arquivo PDB, o programa extrai o sítio de ligação, calcula o tensor de 32 canais, amostra o vetor "
        "latente, constrói as moléculas, minimiza a geometria, verifica colisões com a proteína, ranqueia os "
        "candidatos e grava os resultados.")
    m.figura(f("etapa6.png"),
             "Aba ETAPA 6: alvo e saída, definição do sítio, amostragem, canais, filtros e geometria final.", 14.5)
    m.h2("12.1 Alvo, modelo e saída")
    m.bullets([
        "**Fase do modelo** e **Checkpoints** — modelo a usar; vazio, usa `checkpoints/phase<N>/`. O arquivo "
        "`best_model.pt` precisa existir;",
        "**Proteína (PDB)** — arquivo PDB da proteína-alvo. É o único campo obrigatório;",
        "**Resíduos do sítio (PDB)** — opcional: um PDB contendo apenas os resíduos do sítio, quando o usuário "
        "já sabe exatamente qual é;",
        "**Ligante de ref.** — opcional: o **ligante de referência**, em SDF. Quando fornecido, a similaridade de "
        "Tanimoto de cada candidato com ele entra na pontuação e na tabela de resultados;",
        "**Saída** — pasta dos resultados; vazia, usa `PROJECTS/<projeto>/results/<alvo>/`."])
    m.h2("12.2 Definição do sítio")
    m.p("O sítio pode ser definido de três maneiras, mutuamente exclusivas:")
    m.numerada([
        "**Pelo ligante cocristalizado** — se a proteína tem um ligante (`HETATM`), os resíduos a até 6 Å dele "
        "definem o sítio;",
        "**Por coordenadas do centro** — marque **Definir sítio por coordenadas do centro**, informe X, Y e Z "
        "(no mesmo referencial do PDB) e o **Raio do sítio** (padrão 10 Å). Os resíduos com ao menos um "
        "átomo dentro do raio definem o sítio. Use quando a proteína não tem ligante;",
        "**Por busca de bolsos** — marque **Busca heurística (P2Rank)**. O P2Rank prediz os bolsos e o "
        "escolhido em **Bolso a usar** (1 = maior pontuação) define o sítio. Exige o P2Rank e o Java 17 ou "
        "superior; o ranking completo é gravado em `pockets_p2rank.json`."])
    m.p("Marcar a busca heurística desabilita as coordenadas manuais. Essas duas opções não podem ser combinadas "
        "com um PDB de resíduos do sítio: o programa avisa do conflito e não executa.")
    m.h2("12.3 Amostragem")
    m.bullets([
        "**Nº de amostras** (padrão 200) — quantas moléculas gerar;",
        "**Temperatura** (padrão 1,0) — diversidade da amostragem. Valores altos achatam também a distribuição "
        "de tipos atômicos, e a geração tende a superrepresentar elementos como P, F e S em relação ao "
        "treino; reduzir para cerca de 0,7 aproxima a composição elementar da base;",
        "**Limiar de parada** (padrão 0,5) — confiança que o modelo precisa atingir para encerrar a "
        "molécula. Aumente (0,6 a 0,7) para ligantes maiores e reduza para menores; ao contrário do peso da "
        "parada, não exige retreinar;",
        "**Limiar de ligação** (padrão 0,5) — probabilidade mínima para aceitar uma ligação além da "
        "âncora com o átomo focal. Reduzir favorece o fechamento de anéis, ao risco de ligações espúrias;",
        "**Cutoff do sítio (Å)** (padrão 6,0) — distância usada para recortar os resíduos do sítio quando há "
        "ligante."])
    m.p("Durante a amostragem, uma **guarda estérica** rejeita e refaz o sorteio de qualquer átomo que "
        "colida com os átomos do sítio, e também de átomos posicionados a distâncias incompatíveis com as "
        "ligações vizinhas. Isso reduz de forma expressiva as colisões da molécula com a proteína, sem "
        "necessidade de retreinar o modelo.")
    m.h2("12.4 Canais do sítio")
    m.p("O grupo **Canais do Sítio (devem bater com o treino)** repete as opções de protonação, pH, método dos "
        "canais HOMO/LUMO e base QM da ETAPA 2. Quando o checkpoint escolhido traz a configuração de "
        "voxelização gravada no treinamento, ela sempre prevalece sobre estes campos, com aviso no log se "
        "divergirem; eles só têm efeito para checkpoints antigos, sem essa configuração.")
    m.h2("12.5 Filtros e ranqueamento")
    m.bullets([
        "**Átomos mín. do sítio** (padrão 5) — número mínimo de átomos pesados do sítio para aceitar o alvo;",
        "**Átomos pesados mín.** (padrão 7) — descarta candidatos com menos átomos pesados que isso, depois "
        "do ranqueamento; assim, o Top N significa os N melhores que passam no filtro. Use 1 para desligar;",
        "**Relatório Top N** (padrão 20) — quantos candidatos entram nos arquivos de saída."])
    m.p("Cada candidato recebe uma pontuação composta entre 0 e 1, soma de: 40% do QED, 20% da acessibilidade "
        "sintética (SA score), 15% da regra de Lipinski, 10% da regra de Veber, 5% do atendimento conjunto de "
        "limites de peso molecular, logP, TPSA e ligações rotacionáveis, e 10% da similaridade de Tanimoto com o "
        "ligante de referência, quando fornecido. Candidatos que colidem com o sítio têm a pontuação "
        "multiplicada por um fator que decresce com o número de colisões, de modo que um QED alto não compense "
        "um encaixe impossível.")
    m.h2("12.6 Geometria final")
    m.bullets([
        "**FF de geometria final** — campo de força da minimização final do **complexo** (ligante mais sítio), "
        "feita antes da checagem de colisão: **MMFF/UFF** (RDKit, padrão, sem dependência extra; o UFF cobre "
        "metais como Zn, Fe e Mg), **AMBER ff14SB** ou **CHARMM36** (exigem GROMACS e AmberTools; sem eles o "
        "programa volta ao MMFF/UFF com aviso no log) ou **Nenhum** (geometria crua do decodificador, para "
        "diagnóstico);",
        "**Induced-fit (cadeias laterais flexíveis)** — desmarcado, o sítio é rígido e só o ligante se move; "
        "marcado, apenas o esqueleto (N, Cα, C, O) dos resíduos do sítio é travado e as cadeias laterais "
        "se ajustam ao ligante durante a minimização."])
    m.h2("12.7 Resultados")
    m.p("Ao final, a pasta de saída contém:")
    m.tabela(["Arquivo", "Conteúdo"], [
        ["`candidates.csv`", "Tabela com todos os candidatos e suas métricas: SMILES, pontuação, QED, peso "
                              "molecular, logP, doadores e aceptores de hidrogênio, TPSA, ligações rotacionáveis, "
                              "anéis, regras de Lipinski e Veber, SA score, número de átomos, Tanimoto com o "
                              "ligante de referência, presença e número de colisões com o sítio."],
        ["`top<N>_candidates.sdf`", "Estruturas 3D dos N melhores candidatos."],
        ["`top<N>_candidates.png`", "Grade com as estruturas 2D dos N melhores candidatos."],
        ["`report.html`", "Relatório interativo completo."],
        ["`site_tensor.npz`", "Tensor do sítio usado na geração."],
        ["`site_residues_used.pdb`", "Resíduos do sítio efetivamente usados."],
        ["`pockets_p2rank.json`", "Ranking de bolsos do P2Rank, quando a busca heurística foi usada."],
    ], [4.6, 11.4])
    m.figura(f("top20.png"), "Grade 2D dos melhores candidatos gerados (`top20_candidates.png`).", 11.5)
    m.nota("Os candidatos são propostas de partida geradas por um modelo estatístico. A pontuação e as "
           "métricas orientam a priorização, mas não substituem a validação por acoplamento molecular, "
           "dinâmica molecular e, em última instância, ensaios experimentais.", "Importante:")

    # ====================================================================== 13
    m.h1("13. Gerar Relatório Final")
    m.p("A opção **Ajuda > Gerar Relatório Final** salva o estado de todas as abas do projeto atual no "
        "arquivo `final_report.json`, na pasta do projeto, junto com a versão do programa e a data de "
        "geração. Esse arquivo é a fonte de dados do relatório final do projeto, que ainda está em "
        "desenvolvimento; por ora, ele registra os parâmetros usados em cada etapa e é útil para documentar e "
        "reproduzir uma análise.")

    # ====================================================================== 14
    m.h1("14. Fluxo recomendado")
    m.p("A sequência de abas corresponde ao fluxo recomendado. Um roteiro típico, partindo de um conjunto "
        "externo e especializando o modelo, é:")
    m.numerada([
        "**INÍCIO** — conferir as especificações e instalar as dependências que faltam;",
        "**CONFIG** — criar o projeto e, para as Fases 2 e 3, preencher os termos de busca;",
        "**ETAPA 1** — coletar os dados, curá-los e examinar o painel de exploração;",
        "**ETAPA 2** — voxelizar os pares curados (GFN-xTB, se disponível);",
        "**ETAPA 3** (opcional) — buscar hiperparâmetros e escolher as melhores configurações;",
        "**ETAPA 4** — treinar o modelo, na Fase 1 e depois, se for o caso, nas Fases 2 e 3 por ajuste fino;",
        "**ETAPA 5** — avaliar o modelo, conferir as curvas e executar o diagnóstico de Active Units;",
        "**ETAPA 6** — gerar candidatos para a proteína-alvo, definindo o sítio por ligante, coordenadas ou "
        "P2Rank, e analisar os resultados;",
        "**Ajuda > Gerar Relatório Final** — registrar os parâmetros do projeto."])
    m.nota("Se o objetivo é apenas gerar ligantes para uma proteína nova com um modelo já treinado, as "
           "ETAPAS 1 a 5 podem ser puladas: basta indicar o checkpoint na ETAPA 6.")

    # ====================================================================== 15
    m.h1("15. Licença, repositório e citação")
    m.p("O CODEEP é distribuído sob a licença GNU General Public License, versão 3 ou posterior "
        "(GPL-3.0-or-later). O código-fonte e os tutoriais estão disponíveis em "
        "`https://github.com/moimaian/CODEEP`.")
    m.p("O programa utiliza componentes de terceiros, cada um com sua licença, entre eles o RDKit, o "
        "PyTorch, o PyTorch Geometric, o GFN-xTB e o P2Rank. Ao publicar resultados obtidos com o CODEEP, cite o "
        "programa e sua versão, indicada no rodapé da aba INÍCIO e na janela Sobre. Para o P2Rank, cite "
        "Krivák e Hoksza (2018); para a métrica de unidades ativas, Burda, Grosse e Salakhutdinov (2016).")
