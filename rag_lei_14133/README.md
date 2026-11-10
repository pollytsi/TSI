# Perguntas e respostas sobre a Lei nº 14.133/2021 com RAG

Sistema de perguntas e respostas sobre a [Lei nº 14.133/2021](https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm) (Lei de Licitações e Contratos Administrativos). Ele usa *Retrieval-Augmented Generation* (RAG): o texto da lei é dividido em trechos (*chunks*), cada trecho vira um vetor de *embeddings* guardado em um banco de dados vetorial [FAISS](https://github.com/facebookresearch/faiss), e o módulo `RetrievalQA` do [LangChain](https://www.langchain.com/) busca os trechos mais parecidos com a pergunta e os entrega a um LLM, que redige a resposta citando os artigos.

Tudo roda localmente com o [Ollama](https://ollama.com): nenhum texto ou pergunta sai da máquina.

O projeto é baseado no notebook [vb-faiss-un04.ipynb](https://github.com/tmedeirosb/tsi-llm/blob/main/vb-faiss-un04.ipynb), atualizado para a API atual do LangChain (1.x).

## Como funciona

```
                       criar_indice.py (executado uma vez)
┌───────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐   ┌─────────────┐
│ Site do       │──▶│ Extração do  │──▶│ Divisão em  │──▶│ Embeddings       │──▶│ Índice FAISS│
│ Planalto      │   │ texto vigente│   │ chunks      │   │ nomic-embed-text │   │ (em disco)  │
└───────────────┘   └──────────────┘   └─────────────┘   └──────────────────┘   └──────┬──────┘
                                                                                       │
                       perguntar.py (RetrievalQA)                                      │
┌───────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────┐   │
│ Pergunta do   │──▶│ Embedding da     │──▶│ Busca dos k chunks │◀──┤              │◀──┘
│ usuário       │   │ pergunta         │   │ mais próximos      │   │              │
└───────────────┘   └──────────────────┘   └─────────┬──────────┘   └──────────────┘
                                                     ▼
                                           ┌────────────────────┐   ┌──────────────┐
                                           │ LLM llama3.2 lê os │──▶│ Resposta com │
                                           │ trechos e responde │   │ os artigos   │
                                           └────────────────────┘   └──────────────┘
```

### 1. Criação do banco vetorial — [`criar_indice.py`](criar_indice.py)

1. **Download**: baixa a página da lei do site do Planalto e guarda uma cópia em `data/l14133.htm`.
2. **Extração**: converte o HTML (codificação Windows-1252) em texto, descartando a redação revogada ou alterada, que o Planalto marca com `<strike>` ou com o estilo `text-decoration: line-through`. O texto extraído é salvo em `data/l14133.txt` para conferência.
3. **Divisão em chunks**: um chunk por artigo; artigos com mais de 1.500 caracteres são subdivididos (sobreposição de 150), e cada pedaço começa com o rótulo `[Art. Xº, continuação]` para não perder o contexto. O número do artigo fica nos metadados, o que permite citar a fonte. Há dois casos especiais:
   - os crimes que o art. 178 insere no Código Penal (arts. 337-E a 337-P) viram um chunk cada, rotulados como `Art. 178 (inclui Art. 337-F)`, para que penas de crimes diferentes não se misturem;
   - o art. 6º (definições) usa chunks de 500 caracteres, com poucas definições em cada.
4. **Embeddings**: cada chunk é convertido em um vetor de 768 dimensões pelo modelo `nomic-embed-text`, com o prefixo `search_document:` que o modelo espera.
5. **Índice FAISS**: os vetores são inseridos em um índice FAISS e gravados em `faiss_index/lei_14133/`.

Resultado: **196 artigos → 342 chunks → 342 vetores de 768 dimensões**.

### 2. Perguntas e respostas — [`perguntar.py`](perguntar.py)

1. Recebe a pergunta pela linha de comando ou em modo interativo.
2. Converte a pergunta em embedding (`nomic-embed-text`, prefixo `search_query:`).
3. Busca no FAISS os `k` chunks mais próximos (padrão: 5).
4. O `RetrievalQA` (`chain_type="stuff"`) monta um prompt com os trechos, cada um identificado pelo artigo, e o envia ao `llama3.2` com temperatura 0. O prompt exige resposta baseada apenas nos trechos, com citação dos artigos, e que o modelo diga quando não encontrar a informação.
5. Exibe a resposta e a lista de artigos consultados.

## Requisitos

| Componente | Versão usada nos testes |
|---|---|
| Python | 3.13.3 |
| Ollama | 0.40.13 |
| Modelo de embeddings | `nomic-embed-text` (~275 MB) |
| LLM | `llama3.2` (3B, ~2 GB) |
| Pacotes Python | ver [`requirements.txt`](requirements.txt) (LangChain 1.4.9, FAISS 1.15.1) |

## Instalação

Os comandos abaixo partem da raiz do repositório.

### 1. Clonar o repositório

```bash
git clone https://github.com/pollytsi/TSI.git
cd TSI
```

### 2. Instalar o Ollama e baixar os modelos

Instale o Ollama a partir de <https://ollama.com/download> (no Linux: `curl -fsSL https://ollama.com/install.sh | sh`) e baixe os dois modelos:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

No Windows e no macOS o servidor do Ollama inicia sozinho. Se não estiver ativo, rode `ollama serve` em um terminal separado. Para testar, abra <http://localhost:11434>: deve aparecer `Ollama is running`.

### 3. Criar o ambiente virtual e instalar as dependências

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r rag_lei_14133/requirements.txt
```

## Execução

### 1. Criar o banco vetorial (uma vez)

```bash
python rag_lei_14133/criar_indice.py
```

Saída:

```
Baixando https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm
Texto extraído: 265,704 caracteres (salvo em ...\rag_lei_14133\data\l14133.txt)
Artigos: 196  |  Chunks: 342
Gerando embeddings com 'nomic-embed-text'...
  342/342 chunks
Embeddings gerados em 410.6 s
Índice FAISS: 342 vetores de 768 dimensões
Salvo em ...\rag_lei_14133\faiss_index\lei_14133/
```

A geração dos embeddings levou cerca de 7 minutos no computador dos testes. Nas execuções seguintes o script reaproveita a cópia local da lei; use `--baixar` para baixá-la de novo.

| Opção | Padrão | Descrição |
|---|---|---|
| `--modelo` | `nomic-embed-text` | Modelo de embeddings do Ollama |
| `--tamanho-chunk` | `1500` | Máximo de caracteres por chunk |
| `--sobreposicao` | `150` | Caracteres repetidos entre chunks vizinhos |
| `--baixar` | desativado | Baixa a lei de novo, ignorando a cópia local |

### 2. Fazer perguntas

Uma pergunta só:

```bash
python rag_lei_14133/perguntar.py "Quais são as modalidades de licitação?"
```

Modo interativo (linha vazia ou `sair` encerra):

```bash
python rag_lei_14133/perguntar.py
```

| Opção | Padrão | Descrição |
|---|---|---|
| `pergunta` | — | Pergunta; sem ela, abre o modo interativo |
| `--k` | `5` | Quantidade de trechos recuperados do FAISS |
| `--modelo-llm` | `llama3.2:latest` | Modelo do Ollama que gera a resposta |
| `--modelo-embeddings` | `nomic-embed-text` | Deve ser o mesmo usado na criação do índice |
| `--trechos` | desativado | Exibe também o texto dos trechos recuperados |

Cada resposta levou de 1 a 2 minutos no computador dos testes, onde o `llama3.2` roda em parte na CPU. Com uma GPU que comporte o modelo inteiro, a resposta sai em segundos.

## Exemplos de perguntas e respostas

Respostas reais do sistema com a configuração padrão (`k=5`, `llama3.2`), copiadas sem edição. A linha "Trechos consultados" mostra o número de chunks recuperados e os artigos de origem, sem repetição. Por isso pode listar menos artigos do que chunks.

### ✅ Quais são as modalidades de licitação?

```
Resposta:
Conforme o Art. 28, as modalidades de licitação são:

I - pregão;
II - concorrência;
III - concurso;
IV - leilão;
V - diálogo competitivo.

Além disso, o Art. 6º também menciona outras modalidades de licitação, como:

- leilão: modalidade de licitação para alienação de bens imóveis ou de bens móveis inservíveis ou legalmente apreendidos;
- pregão: modalidade de licitação obrigatória para aquisição de bens e serviços comuns, cujo critério de julgamento poderá ser o de menor preço ou o de maior desconto;
- diálogo competitivo: modalidade de licitação para contratação de obras.

Portanto, as modalidades de licitação mencionadas na Lei nº 14.133/2021 são:

I - pregão;
II - concorrência;
III - concurso;
IV - leilão;
V - diálogo competitivo.

Trechos consultados (5): Art. 28, Art. 6º, Art. 62, Art. 74
```

Correta: o art. 28 lista exatamente essas cinco modalidades. O complemento tirado do art. 6º é redundante, e a definição de diálogo competitivo foi resumida demais (a lei fala em "obras, serviços e compras").

### ✅ Quais são as fases do processo de licitação?

```
Resposta:
Conforme o Art. 17, as fases do processo de licitação são as seguintes:

I - preparatória;
II - de divulgação do edital de licitação;
III - de apresentação de propostas e lances, quando for o caso;
IV - de julgamento;
V - de habilitação;
VI - de recursal;
VII - de homologação.

Essas fases são mencionadas no Art. 17 da Lei nº 14.133/2021.

Trechos consultados (5): Art. 17, Art. 12, Art. 178 (inclui Art. 337-J), Art. 15
```

Correta e na ordem da lei.

### ✅ Qual a pena para quem frustra o caráter competitivo de uma licitação?

```
Resposta:
Conforme o Art. 337-F, a pena para quem frustra o caráter competitivo de uma licitação é reclusão, de 4 (quatro) anos a 8 (oito) anos, e multa.

Trechos consultados (5): Art. 178 (inclui Art. 337-F), Art. 74, Art. 6º, Art. 178 (inclui Art. 337-K), Art. 180
```

Correta. O art. 337-F foi incluído no Código Penal pelo art. 178 da lei. Antes de cada crime virar um chunk separado, o sistema respondia com a pena de outro crime (art. 337-I).

### ✅ Qual o valor para ser considerado grande vulto?

```
Resposta:
Conforme o Art. 6º, XXII, o valor para ser considerado grande vulto é de R$ 200.000.000,00 (duzentos milhões de reais).

Trechos consultados (5): Art. 6º, Art. 99, Art. 67, Art. 59
```

Correta segundo o texto da lei. O valor é atualizado anualmente por decreto, e o decreto não faz parte do índice (veja as [Limitações](#limitações)).

### ❌ O que é contratação de grande vulto?

```
Resposta:
Conforme o Art. 99 da Lei nº 14.133/2021, a contratação de obras e serviços de engenharia de grande vulto é definida como aquela que pode exigir a prestação de garantia, na modalidade seguro-garantia, com cláusula de retomada prevista no art. 102 desta Lei, em percentual equivalente a até 30% (trinta por cento) do valor inicial do contrato.

Trechos consultados (5): Art. 99, Art. 118, Art. 178 (inclui Art. 337-E), Art. 128, Art. 73
```

**Errada.** É a mesma dúvida do exemplo anterior, com outras palavras. Desta vez a definição (art. 6º, XXII) ficou em 8º lugar na busca, fora dos 5 trechos enviados ao LLM. O art. 99 contém a expressão "contratações de obras e serviços de engenharia de grande vulto" e ficou mais próximo da pergunta, e o modelo apresentou uma regra de garantia como se fosse a definição.

### ✅ Qual é a capital da França? (pergunta fora do assunto)

```
Resposta:
Não encontrou a informação na lei. A Lei nº 14.133/2021 trata de licitações e contratos administrativos, e não aborda a geografia política de países estrangeiros.

Trechos consultados (5): Art. 157, Art. 182, Art. 6º, Art. 124, Art. 69
```

Correta: o modelo seguiu a instrução de não responder com conhecimento de fora da lei.

## Limitações

- **Busca só por similaridade semântica.** Quando a pergunta usa as mesmas palavras de outro artigo, o trecho certo pode ficar fora dos `k` recuperados (veja o exemplo de "grande vulto"). Aumentar `--k` ou reformular a pergunta ajuda; uma busca híbrida (FAISS + BM25 por palavras-chave) seria a melhoria natural.
- **Artigos longos.** Um artigo dividido em muitos chunks (como o art. 75, com as hipóteses de dispensa de licitação) pode ter só parte dos pedaços recuperada, e a resposta sai incompleta.
- **Valores atualizados por decreto.** O índice contém apenas o texto da lei. Os valores que a lei manda atualizar por decreto (limites de dispensa, grande vulto etc.) aparecem com o valor original.
- **Modelo pequeno.** O `llama3.2` (3B) às vezes cita o dispositivo errado ou resume demais. Modelos maiores do Ollama podem ser usados com `--modelo-llm`.
- **Não é aconselhamento jurídico.** As respostas são geradas automaticamente e devem ser conferidas no texto oficial da lei.

## Solução de problemas

| Mensagem | Causa provável | Como resolver |
|---|---|---|
| `Índice não encontrado em ...` | O banco vetorial ainda não foi criado | Rode `python rag_lei_14133/criar_indice.py` |
| `Erro ao baixar a lei: ... Connection reset` | O site do Planalto recusou a conexão | Tente de novo mais tarde; o script já envia cabeçalhos de navegador |
| `model "..." not found` | Modelo não baixado no Ollama | `ollama pull nomic-embed-text` e `ollama pull llama3.2` |
| `Erro ao conectar ao Ollama` / `Connection refused` | Servidor Ollama parado | Rode `ollama serve` |
| Respostas muito lentas | LLM rodando na CPU | Normal sem GPU; reduza `--k` para enviar menos texto ao LLM |

## Estrutura

```
rag_lei_14133/
├── criar_indice.py      # baixa a lei, divide em chunks, gera embeddings e cria o índice FAISS
├── perguntar.py         # pipeline RetrievalQA: pergunta → embedding → busca → resposta
├── requirements.txt     # dependências Python
├── README.md            # este arquivo
├── data/                # gerado: cópia da lei (HTML e texto extraído)
└── faiss_index/         # gerado: índice FAISS (index.faiss e index.pkl)
```

As pastas `data/` e `faiss_index/` são geradas por `criar_indice.py` e não são versionadas. O índice inclui um arquivo `pickle`, que só deve ser carregado quando foi gerado na própria máquina; por isso o repositório não distribui um índice pronto.
