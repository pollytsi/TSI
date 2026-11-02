# TSI — Embeddings de texto com Ollama

Script em Python que gera o vetor de *embeddings* de um texto usando um modelo executado localmente pelo [Ollama](https://ollama.com). Nenhum dado sai da máquina: o texto é enviado apenas para o servidor Ollama local.

O script recebe uma frase, pede ao modelo `nomic-embed-text` o vetor correspondente (768 dimensões) e exibe um resumo: tamanho, norma, valores mínimo e máximo e os primeiros valores do vetor.

## Requisitos

| Componente | Versão usada nos testes | Observação |
|---|---|---|
| Python | 3.13.3 | Qualquer Python 3.8+ deve funcionar |
| Ollama | 0.40.13 | Precisa estar em execução durante o uso |
| Modelo `nomic-embed-text` | `latest` | Cerca de 275 MB de download |

O script usa **somente a biblioteca padrão do Python**. Não há pacotes para instalar com `pip`.

## Reprodução do ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/pollytsi/TSI.git
cd TSI
```

### 2. Instalar o Python

Baixe em <https://www.python.org/downloads/> e, no Windows, marque a opção **"Add python.exe to PATH"** durante a instalação. Confira:

```bash
python --version
```

> Em Linux e macOS o comando pode ser `python3`.

### 3. Instalar o Ollama

- **Windows e macOS:** baixe o instalador em <https://ollama.com/download> e execute-o.
- **Linux:**

  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Confira a instalação:

```bash
ollama --version
```

### 4. Baixar o modelo de embeddings

```bash
ollama pull nomic-embed-text
```

Para ver os modelos instalados:

```bash
ollama list
```

### 5. Garantir que o servidor Ollama está rodando

No Windows e no macOS, o Ollama inicia sozinho em segundo plano após a instalação (ícone na bandeja do sistema). Se não estiver ativo, ou no Linux sem serviço configurado, inicie manualmente em um terminal separado:

```bash
ollama serve
```

O servidor responde em `http://localhost:11434`. Para testar, abra esse endereço no navegador: deve aparecer a mensagem `Ollama is running`.

### 6. (Opcional) Ambiente virtual

Como o script não tem dependências externas, o ambiente virtual não é obrigatório. Se quiser isolar o projeto mesmo assim:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

## Execução

Com o Ollama em execução, rode a partir da pasta do repositório:

```bash
python ollama_embeddings.py
```

Sem argumentos, o script usa a frase padrão `"O céu está azul e o dia está lindo."`. Exemplo de saída:

```
Conectado ao Ollama em http://localhost:11434
Modelos instalados: nomic-embed-text:latest

Texto:   O céu está azul e o dia está lindo.
Modelo:  nomic-embed-text
Tamanho: 768 dimensões
Norma:   0.999999
Mínimo:  -0.137512
Máximo:  0.124175
Primeiros 10 valores: [0.014525, 0.029620, -0.120042, 0.000238, 0.036691, -0.015764, -0.033071, 0.000168, -0.025600, -0.040104]
```

### Opções

| Argumento | Padrão | Descrição |
|---|---|---|
| `texto` | `"O céu está azul e o dia está lindo."` | Texto de entrada (use aspas se tiver espaços) |
| `--modelo` | `nomic-embed-text` | Modelo do Ollama usado para gerar o vetor |
| `--host` | `http://localhost:11434` | Endereço do servidor Ollama |
| `--completo` | desativado | Imprime o vetor inteiro, além do resumo |

Exemplos:

```bash
# Outro texto
python ollama_embeddings.py "Aprendizado de máquina é fascinante"

# Mostrar o vetor completo
python ollama_embeddings.py "Aprendizado de máquina é fascinante" --completo

# Servidor Ollama em outra máquina da rede
python ollama_embeddings.py --host 192.168.0.10:11434
```

O endereço do servidor também pode ser definido pela variável de ambiente `OLLAMA_HOST`, que é usada quando `--host` não é informado.

## Solução de problemas

| Mensagem | Causa provável | Como resolver |
|---|---|---|
| `Não foi possível conectar ao Ollama em ...` | Servidor Ollama parado ou em outro endereço | Execute `ollama serve` ou confira o valor de `--host` / `OLLAMA_HOST` |
| `HTTP 404: {"error":"model 'nomic-embed-text' not found"}` | Modelo não baixado | Execute `ollama pull nomic-embed-text` |
| `O modelo '...' não retornou embeddings.` | O modelo escolhido não gera embeddings | Use um modelo de embeddings, como `nomic-embed-text` |
| `python` não é reconhecido | Python fora do `PATH` | Reinstale marcando "Add python.exe to PATH" ou use `python3` / `py` |

## Estrutura do repositório

```
TSI/
├── ollama_embeddings.py   # script que gera e exibe o embedding
└── README.md              # este arquivo
```
