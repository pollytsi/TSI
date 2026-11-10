"""Responde perguntas sobre a Lei nº 14.133/2021 com RetrievalQA (LangChain + FAISS).

Pipeline:
    1. Recebe a pergunta do usuário.
    2. Converte a pergunta em embedding (nomic-embed-text, prefixo "search_query:").
    3. Busca no índice FAISS os k chunks mais próximos desse embedding.
    4. Envia os chunks e a pergunta ao LLM (llama3.2), que responde com base neles.

Uso:
    python perguntar.py "Quais são as modalidades de licitação?"
    python perguntar.py --k 8 --trechos "Quando a licitação é dispensável?"
    python perguntar.py          (modo interativo; linha vazia ou "sair" encerra)

Requer o índice criado por criar_indice.py e o Ollama em execução com os
modelos nomic-embed-text e llama3.2.
"""

import argparse
import sys
import warnings

# O LangChain avisa que o langchain-community e o RetrievalQA estão em descontinuação;
# os avisos poluem a saída do terminal e não afetam o funcionamento.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from criar_indice import MODELO_PADRAO, PASTA_INDICE, OllamaEmbeddingsNomic

MODELO_LLM = "llama3.2:latest"
K_PADRAO = 5
# Tokens de contexto do LLM: cabem os k trechos (~500 tokens cada), o prompt e a resposta.
# Valores maiores ocupam mais memória e tiram parte do modelo da GPU, deixando-o lento.
TAMANHO_CONTEXTO = 4096

PROMPT = PromptTemplate.from_template(
    """Você é um assistente especializado na Lei nº 14.133/2021 (Lei de Licitações e Contratos Administrativos).
Responda à pergunta usando SOMENTE os trechos da lei abaixo.
Cite os artigos que fundamentam a resposta (por exemplo: "conforme o Art. 28").
Se os trechos não contiverem a resposta, diga que não encontrou a informação na lei; não invente.
Responda em português, de forma clara e objetiva.

Trechos da lei:
{context}

Pergunta: {question}

Resposta:"""
)

# Cada trecho chega ao LLM identificado pelo artigo de origem, para que ele possa citá-lo.
PROMPT_DOCUMENTO = PromptTemplate.from_template("[{artigo}]\n{page_content}")


def carregar_banco(modelo_embeddings):
    """Carrega o índice FAISS salvo por criar_indice.py."""
    if not (PASTA_INDICE / "index.faiss").exists():
        raise FileNotFoundError(
            f"Índice não encontrado em {PASTA_INDICE}/. Rode antes: python criar_indice.py"
        )
    # O index.pkl é gerado localmente por criar_indice.py, por isso a desserialização é segura.
    return FAISS.load_local(
        str(PASTA_INDICE),
        OllamaEmbeddingsNomic(model=modelo_embeddings),
        allow_dangerous_deserialization=True,
    )


def criar_cadeia(banco, modelo_llm, k):
    """Conecta o RetrievalQA ao banco vetorial e ao LLM."""
    retriever = banco.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = OllamaLLM(model=modelo_llm, temperature=0, num_ctx=TAMANHO_CONTEXTO)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT, "document_prompt": PROMPT_DOCUMENTO},
    )


def responder(cadeia, pergunta, mostrar_trechos=False):
    """Executa o pipeline para uma pergunta e exibe a resposta e as fontes."""
    resultado = cadeia.invoke({"query": pergunta})
    fontes = resultado["source_documents"]

    print(f"\nResposta:\n{resultado['result'].strip()}\n")
    artigos = list(dict.fromkeys(doc.metadata["artigo"] for doc in fontes))
    print(f"Trechos consultados ({len(fontes)}): {', '.join(artigos)}")

    if mostrar_trechos:
        for posicao, doc in enumerate(fontes, start=1):
            print(f"\n--- {posicao}. {doc.metadata['artigo']} (parte {doc.metadata['parte']}) ---")
            print(doc.page_content)


def main():
    parser = argparse.ArgumentParser(
        description="Perguntas sobre a Lei nº 14.133/2021 com RetrievalQA."
    )
    parser.add_argument("pergunta", nargs="?", help="pergunta; sem ela, abre o modo interativo")
    parser.add_argument("--k", type=int, default=K_PADRAO, help="quantidade de trechos recuperados")
    parser.add_argument("--modelo-llm", default=MODELO_LLM, help="modelo do Ollama que gera a resposta")
    parser.add_argument("--modelo-embeddings", default=MODELO_PADRAO, help="modelo usado na criação do índice")
    parser.add_argument("--trechos", action="store_true", help="exibe o texto dos trechos recuperados")
    args = parser.parse_args()

    # Evita erro de codificação ao imprimir acentos em consoles antigos do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    try:
        banco = carregar_banco(args.modelo_embeddings)
    except FileNotFoundError as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    cadeia = criar_cadeia(banco, args.modelo_llm, args.k)
    print(f"Índice carregado: {banco.index.ntotal} trechos | LLM: {args.modelo_llm} | k={args.k}")

    try:
        if args.pergunta:
            responder(cadeia, args.pergunta, args.trechos)
            return 0

        print("Faça perguntas sobre a Lei nº 14.133/2021 (linha vazia ou 'sair' encerra).")
        while True:
            pergunta = input("\nPergunta: ").strip()
            if pergunta.lower() in {"", "sair"}:
                return 0
            responder(cadeia, pergunta, args.trechos)
    except (KeyboardInterrupt, EOFError):
        print()
        return 0
    except ConnectionError as erro:
        print(f"Erro ao conectar ao Ollama: {erro}. Verifique se ele está em execução.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
