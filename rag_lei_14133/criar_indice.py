"""Cria o banco de dados vetorial (FAISS) da Lei nº 14.133/2021.

Etapas:
    1. Baixa o texto da lei do site do Planalto (com cópia local em data/).
    2. Extrai o texto vigente do HTML, descartando os trechos revogados (<strike>).
    3. Divide o texto em chunks, um ou mais por artigo.
    4. Gera os embeddings de cada chunk com o Ollama local.
    5. Insere os embeddings em um índice FAISS e o salva em disco.

Uso:
    python criar_indice.py
    python criar_indice.py --tamanho-chunk 1000 --sobreposicao 100 --baixar

Requer o Ollama em execução e o modelo: ollama pull nomic-embed-text
"""

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
import warnings
from html.parser import HTMLParser
from pathlib import Path

# O LangChain avisa que o langchain-community está em descontinuação; o aviso
# polui a saída do terminal e não afeta o funcionamento.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

URL_LEI = "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm"
PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_DADOS = PASTA_PROJETO / "data"
ARQUIVO_HTML = PASTA_DADOS / "l14133.htm"
ARQUIVO_TEXTO = PASTA_DADOS / "l14133.txt"
PASTA_INDICE = PASTA_PROJETO / "faiss_index" / "lei_14133"
MODELO_PADRAO = "nomic-embed-text"
TAMANHO_CHUNK = 1500
SOBREPOSICAO = 150
TAMANHO_CHUNK_DEFINICOES = 500
LOTE_EMBEDDINGS = 64

# O servidor do Planalto derruba conexões que não parecem vir de um navegador.
CABECALHOS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9",
}

# Marca o fim do texto da lei; depois dela vem a promulgação das partes vetadas,
# que já está incorporada ao corpo da lei.
FIM_DA_LEI = "Este texto não substitui"
INICIO_ARTIGO = re.compile(r"^Art\.\s*\d+[º°o]?(?:-[A-Z])?", re.MULTILINE)
SALTO_MAXIMO_ARTIGO = 3


class OllamaEmbeddingsNomic(OllamaEmbeddings):
    """Embeddings do nomic-embed-text com os prefixos de tarefa que o modelo espera.

    O nomic-embed-text foi treinado para distinguir documentos ("search_document:")
    de perguntas ("search_query:"); sem os prefixos a busca fica menos precisa.
    """

    def embed_documents(self, texts):
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"search_query: {text}")


class ExtratorTexto(HTMLParser):
    """Converte o HTML do Planalto em texto, uma linha por parágrafo.

    O Planalto marca o texto revogado ou alterado de duas formas: com <strike>
    e com o estilo "text-decoration: line-through". Ambos são descartados.
    """

    IGNORAR = {"script", "style", "strike", "s", "del", "head"}
    BLOCOS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.partes = []
        # Cada item é [tag que iniciou o trecho ignorado, tags iguais abertas dentro dele].
        self.ignorados = []

    def _deve_ignorar(self, tag, attrs):
        estilo = (dict(attrs).get("style") or "").replace(" ", "").lower()
        return tag in self.IGNORAR or "text-decoration:line-through" in estilo

    def handle_starttag(self, tag, attrs):
        if self._deve_ignorar(tag, attrs):
            self.ignorados.append([tag, 0])
        elif self.ignorados and tag == self.ignorados[-1][0]:
            self.ignorados[-1][1] += 1
        if tag in self.BLOCOS:
            self.partes.append("\n")

    def handle_endtag(self, tag):
        if self.ignorados and tag == self.ignorados[-1][0]:
            if self.ignorados[-1][1]:
                self.ignorados[-1][1] -= 1
            else:
                self.ignorados.pop()
        if tag in self.BLOCOS:
            self.partes.append("\n")

    def handle_data(self, data):
        if not self.ignorados:
            self.partes.append(data)

    def texto(self):
        linhas = (" ".join(linha.split()) for linha in "".join(self.partes).split("\n"))
        return "\n".join(linha for linha in linhas if linha)


def baixar_html(forcar=False):
    """Devolve o HTML da lei, baixando-o apenas se não houver cópia local."""
    if ARQUIVO_HTML.exists() and not forcar:
        print(f"Usando cópia local: {ARQUIVO_HTML}")
        return ARQUIVO_HTML.read_bytes().decode("cp1252")

    print(f"Baixando {URL_LEI}")
    requisicao = urllib.request.Request(URL_LEI, headers=CABECALHOS)
    with urllib.request.urlopen(requisicao, timeout=120) as resposta:
        conteudo = resposta.read()
    PASTA_DADOS.mkdir(exist_ok=True)
    ARQUIVO_HTML.write_bytes(conteudo)
    # A página do Planalto é servida em Windows-1252.
    return conteudo.decode("cp1252")


def extrair_texto(html):
    """Extrai o texto vigente da lei, sem os trechos revogados nem o rodapé."""
    extrator = ExtratorTexto()
    extrator.feed(html)
    texto = extrator.texto()
    fim = texto.find(FIM_DA_LEI)
    if fim == -1:
        raise ValueError("Fim do texto da lei não encontrado; o HTML mudou de formato?")
    return texto[:fim].strip()


def _rotulo(trecho):
    """Devolve "Art. 6º" para um trecho que começa com "Art.\\n6º. ..."."""
    return "Art. " + INICIO_ARTIGO.match(trecho).group(0).replace("Art.", "").strip()


def _inicio_com_titulo(texto, posicao):
    """Recua até o início do título do artigo (ex.: nome do crime), que pode ter várias linhas.

    Títulos não terminam em pontuação; o fim de um dispositivo, sim.
    """
    while posicao > 0:
        inicio_linha = texto.rfind("\n", 0, posicao - 1) + 1
        anterior = texto[inicio_linha:posicao].strip()
        if not anterior or anterior[-1] in ".;:)”":
            break
        posicao = inicio_linha
    return posicao


def dividir_em_chunks(texto, tamanho, sobreposicao):
    """Divide a lei por artigo e subdivide os artigos maiores que `tamanho`.

    Cada chunk guarda o artigo de origem nos metadados. Os pedaços de um artigo
    longo recebem o rótulo "[Art. Xº, continuação]" para não perder o contexto.

    Casos especiais:
    - Artigos que esta lei insere em outras leis (como os arts. 337-E a 337-P do
      Código Penal, incluídos pelo art. 178) não são artigos desta lei: só abre
      um novo artigo o número igual ao anterior (ex.: 44-A) ou pouco maior. Cada
      artigo inserido vira uma unidade própria, rotulada "Art. 178 (inclui Art. 337-F)",
      para que crimes e penas diferentes não se misturem no mesmo chunk.
    - O art. 6º (definições) usa chunks menores, com poucas definições cada.
    """
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=tamanho,
        chunk_overlap=sobreposicao,
        separators=["\n", "; ", ". ", " ", ""],
    )
    divisor_definicoes = RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_CHUNK_DEFINICOES,
        chunk_overlap=sobreposicao // 3,
        separators=["\n", "; ", ". ", " ", ""],
    )

    inicios, inseridos, ultimo = [], [], 0
    for m in INICIO_ARTIGO.finditer(texto):
        numero = int(re.search(r"\d+", m.group(0)).group(0))
        if ultimo <= numero <= ultimo + SALTO_MAXIMO_ARTIGO:
            inicios.append(m.start())
            ultimo = numero
        else:
            inseridos.append(m.start())

    # Unidades de texto: (artigo desta lei, rótulo da unidade, texto).
    unidades = [("Preâmbulo", "Preâmbulo", texto[: inicios[0]])]
    for inicio, fim in zip(inicios, inicios[1:] + [len(texto)]):
        artigo = _rotulo(texto[inicio:fim])
        internos = [p for p in inseridos if inicio < p < fim]
        limites = [inicio] + [_inicio_com_titulo(texto, p) for p in internos] + [fim]
        rotulos = [artigo] + [f"{artigo} (inclui {_rotulo(texto[p:])})" for p in internos]
        for rotulo, a, b in zip(rotulos, limites, limites[1:]):
            unidades.append((artigo, rotulo, texto[a:b]))

    chunks = []
    for artigo, rotulo, trecho in unidades:
        separador = divisor_definicoes if artigo == "Art. 6º" else divisor
        for parte, conteudo in enumerate(separador.split_text(trecho.strip())):
            if parte > 0 or rotulo != artigo:
                conteudo = f"[{rotulo}{', continuação' if parte else ''}]\n{conteudo}"
            chunks.append(
                Document(
                    page_content=conteudo,
                    metadata={"fonte": URL_LEI, "artigo": rotulo, "parte": parte + 1},
                )
            )
    return chunks, len(inicios)


def criar_indice(chunks, embeddings):
    """Gera os embeddings em lotes e os insere em um índice FAISS."""
    banco = None
    for inicio in range(0, len(chunks), LOTE_EMBEDDINGS):
        lote = chunks[inicio : inicio + LOTE_EMBEDDINGS]
        if banco is None:
            banco = FAISS.from_documents(lote, embeddings)
        else:
            banco.add_documents(lote)
        print(f"  {min(inicio + LOTE_EMBEDDINGS, len(chunks))}/{len(chunks)} chunks", end="\r")
    print()
    return banco


def main():
    parser = argparse.ArgumentParser(
        description="Cria o banco vetorial FAISS da Lei nº 14.133/2021."
    )
    parser.add_argument("--modelo", default=MODELO_PADRAO, help="modelo de embeddings do Ollama")
    parser.add_argument("--tamanho-chunk", type=int, default=TAMANHO_CHUNK, help="máximo de caracteres por chunk")
    parser.add_argument("--sobreposicao", type=int, default=SOBREPOSICAO, help="caracteres repetidos entre chunks")
    parser.add_argument("--baixar", action="store_true", help="baixa a lei de novo, ignorando a cópia local")
    args = parser.parse_args()

    # Evita erro de codificação ao imprimir acentos em consoles antigos do Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        html = baixar_html(args.baixar)
    except (urllib.error.URLError, TimeoutError) as erro:
        print(f"Erro ao baixar a lei: {erro}", file=sys.stderr)
        return 1

    texto = extrair_texto(html)
    ARQUIVO_TEXTO.write_text(texto, encoding="utf-8")
    print(f"Texto extraído: {len(texto):,} caracteres (salvo em {ARQUIVO_TEXTO})")

    chunks, artigos = dividir_em_chunks(texto, args.tamanho_chunk, args.sobreposicao)
    print(f"Artigos: {artigos}  |  Chunks: {len(chunks)}")

    print(f"Gerando embeddings com '{args.modelo}'...")
    inicio = time.perf_counter()
    banco = criar_indice(chunks, OllamaEmbeddingsNomic(model=args.modelo))
    print(f"Embeddings gerados em {time.perf_counter() - inicio:.1f} s")

    banco.save_local(str(PASTA_INDICE))
    print(f"Índice FAISS: {banco.index.ntotal} vetores de {banco.index.d} dimensões")
    print(f"Salvo em {PASTA_INDICE}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
