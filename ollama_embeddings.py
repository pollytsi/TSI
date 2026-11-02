"""Gera e exibe o vetor de embeddings de um texto usando o Ollama local.

Uso:
    python ollama_embeddings.py
    python ollama_embeddings.py "Outra frase" --modelo nomic-embed-text --completo

Requer um modelo com suporte a embeddings: ollama pull nomic-embed-text
"""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

HOST_PADRAO = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODELO_PADRAO = "nomic-embed-text"
TEXTO_PADRAO = "O céu está azul e o dia está lindo."
VALORES_INICIAIS = 10


class ErroOllama(Exception):
    """Falha ao se comunicar com o Ollama ou resposta inesperada."""


def _requisitar(url, payload=None, timeout=120):
    """Faz um GET (ou POST, se houver payload) e devolve o JSON da resposta."""
    dados = None if payload is None else json.dumps(payload).encode("utf-8")
    requisicao = urllib.request.Request(
        url, data=dados, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=timeout) as resposta:
            return json.load(resposta)
    except urllib.error.HTTPError as erro:
        detalhe = erro.read().decode("utf-8", errors="replace")
        raise ErroOllama(f"HTTP {erro.code}: {detalhe}") from erro
    except (urllib.error.URLError, TimeoutError) as erro:
        raise ErroOllama(
            f"Não foi possível conectar ao Ollama em {url}. "
            f"Verifique se ele está em execução ('ollama serve'). Detalhe: {erro}"
        ) from erro


def conectar(host):
    """Confirma que o servidor responde e devolve os modelos instalados."""
    resposta = _requisitar(f"{host}/api/tags", timeout=10)
    return [modelo["name"] for modelo in resposta.get("models", [])]


def gerar_embedding(host, modelo, texto):
    """Devolve o vetor de embeddings do texto."""
    resposta = _requisitar(f"{host}/api/embed", {"model": modelo, "input": texto})
    embeddings = resposta.get("embeddings")
    if not embeddings or not embeddings[0]:
        raise ErroOllama(f"O modelo '{modelo}' não retornou embeddings.")
    return embeddings[0]


def exibir(texto, modelo, vetor, completo=False):
    """Mostra o sumário do vetor e, se solicitado, o vetor integral."""
    norma = math.sqrt(sum(valor * valor for valor in vetor))
    iniciais = ", ".join(f"{valor:.6f}" for valor in vetor[:VALORES_INICIAIS])

    print(f"Texto:   {texto}")
    print(f"Modelo:  {modelo}")
    print(f"Tamanho: {len(vetor)} dimensões")
    print(f"Norma:   {norma:.6f}")
    print(f"Mínimo:  {min(vetor):.6f}")
    print(f"Máximo:  {max(vetor):.6f}")
    print(f"Primeiros {min(VALORES_INICIAIS, len(vetor))} valores: [{iniciais}]")

    if completo:
        print("\nVetor completo:")
        print(vetor)


def main():
    parser = argparse.ArgumentParser(
        description="Gera embeddings de um texto com o Ollama local."
    )
    parser.add_argument("texto", nargs="?", default=TEXTO_PADRAO, help="texto de entrada")
    parser.add_argument("--modelo", default=MODELO_PADRAO, help="modelo do Ollama")
    parser.add_argument("--host", default=HOST_PADRAO, help="endereço do servidor Ollama")
    parser.add_argument("--completo", action="store_true", help="exibe o vetor inteiro")
    args = parser.parse_args()

    # Evita erro de codificação ao imprimir acentos em consoles antigos do Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    host = args.host if "://" in args.host else f"http://{args.host}"
    host = host.rstrip("/")

    try:
        modelos = conectar(host)
        print(f"Conectado ao Ollama em {host}")
        print(f"Modelos instalados: {', '.join(modelos) or 'nenhum'}\n")
        vetor = gerar_embedding(host, args.modelo, args.texto)
    except ErroOllama as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1

    exibir(args.texto, args.modelo, vetor, args.completo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
