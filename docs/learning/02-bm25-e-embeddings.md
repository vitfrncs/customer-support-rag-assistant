# Recuperação lexical e semântica: BM25 e embeddings

O objetivo da recuperação de informação é encontrar, em uma coleção de textos, os documentos mais relacionados a uma consulta. Neste projeto, a coleção contém solicitações sintéticas de atendimento do Bitext, e cada busca retorna três exemplos semelhantes à mensagem recebida (`top_k=3`).

Duas abordagens serão comparadas: BM25, que utiliza correspondências entre termos, e E5, que utiliza embeddings. Ambas produzem um ranking, mas calculam a relevância de maneiras diferentes.

## BM25

O BM25 é um algoritmo de recuperação lexical. Para ordenar os documentos, considera quais palavras da consulta aparecem em cada texto, a frequência dessas palavras e o tamanho do documento.

A ideia não é simplesmente contar palavras iguais. Uma palavra presente em quase todos os textos ajuda pouco a identificar um documento específico; uma palavra mais rara costuma ser mais informativa. Além disso, repetir uma palavra várias vezes não aumenta sua importância indefinidamente.

Uma forma comum da fórmula é:

$$
BM25(q,d) = \sum_{t \in q} IDF(t) \cdot
\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1\left(1-b+b\frac{|d|}{avgdl}\right)}
$$

Nessa expressão, `q` é a consulta e `d` é o documento. `tf(t,d)` conta as ocorrências do termo `t` no documento, enquanto `IDF(t)` atribui menos peso a termos frequentes na coleção. `|d|` é o tamanho do documento e `avgdl` é o tamanho médio dos documentos. O parâmetro `k1` controla o efeito da repetição dos termos; `b`, o efeito do comprimento do documento.

No projeto, a implementação utiliza `BM25Okapi`, da biblioteca `rank-bm25`. Antes de indexar, os textos são convertidos para minúsculas e separados por uma expressão regular:

```python
import re

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
```

Essa tokenização não faz correção ortográfica, *stemming* ou lematização. Assim, palavras como `close` e `closing` são consideradas termos distintos. Esse pré-processamento simples faz parte da configuração do baseline e pode influenciar os resultados.

A biblioteca também não realiza esse pré-processamento automaticamente. Os parâmetros padrão de `BM25Okapi` são `k1=1.5`, `b=0.75` e `epsilon=0.25`; o último é usado no tratamento de valores negativos de IDF nessa implementação.

## Embeddings e E5

Embeddings são representações numéricas de textos. Em vez de comparar diretamente as palavras, um modelo de embeddings transforma uma frase em um vetor que pode ser comparado com vetores de outras frases. Modelos treinados para recuperação procuram aproximar textos relacionados nesse espaço vetorial.

O recuperador semântico do projeto usa `intfloat/multilingual-e5-small`, por meio da biblioteca Sentence Transformers. Cada texto gera um embedding com 384 dimensões. Para preparar as entradas, são utilizados os prefixos recomendados pelo modelo:

```python
f"passage: {instruction}"   # exemplos indexados
f"query: {message}"         # consulta
```

Os embeddings são normalizados antes do cálculo da similaridade. A similaridade por cosseno é dada por:

$$
\cos(\theta) = \frac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\|\,\|\mathbf{d}\|}
$$

Como os vetores têm norma 1, o cálculo se reduz ao produto escalar:

$$
\cos(\theta) = \mathbf{q}\cdot\mathbf{d}
$$

O índice armazena os vetores dos 675 exemplos. Quando uma consulta chega, seu embedding é gerado e comparado com os vetores armazenados; os três mais próximos são retornados.

A similaridade não representa a probabilidade de a intenção estar correta. Uma pontuação alta indica proximidade vetorial, mas não garante que os dois textos expressem a mesma ação.

O E5 também possui um limite de comprimento de entrada de 512 tokens. Se a sequência ultrapassar esse valor, parte do texto pode ser truncada. É necessário medir os comprimentos usando o tokenizer do próprio modelo para verificar se isso ocorre nos dados indexados e nas consultas de avaliação.

A quantidade de tokens da entrada não deve ser confundida com as 384 dimensões do embedding de saída.

## Diferenças e avaliação

O BM25 depende da correspondência lexical e pode se beneficiar de termos específicos, como `close` ou `refund`. Por outro lado, pode ter dificuldade quando consulta e documento usam palavras diferentes para expressar a mesma ideia.

O E5 pode aproximar essas formulações, mas também pode confundir ações relacionadas.

Na avaliação inicial do E5, algumas solicitações `get_refund` recuperaram exemplos `check_refund_policy`. Os textos pertencem ao mesmo assunto, mas solicitar um reembolso é diferente de consultar as condições para recebê-lo. Esse tipo de erro importa porque o pipeline utiliza as intenções dos exemplos recuperados para selecionar orientações de atendimento.

A comparação entre BM25 e E5 deve manter os mesmos 675 exemplos, as mesmas 135 consultas e `top_k=3`. As métricas utilizadas são Precision@3, Recall@3, Hit Rate@3 e MRR@3, com relevância definida pela igualdade entre as intenções.

Os resultados iniciais do E5 e as limitações dessa definição estão registrados em `01-retrieval-evaluation.md`.

Os scores dos dois métodos não podem ser comparados diretamente, pois possuem escalas diferentes. A comparação deve considerar as métricas de ranking e os exemplos em que cada método acerta ou erra.


## Referências

- [Keyword Searching Algorithms for Search Engines (BM25)](https://www.geeksforgeeks.org/techtips/keyword-searching-algorithms-for-search-engines)
- [What is Text Embedding?](https://www.geeksforgeeks.org/nlp/what-is-text-embedding)
- [Sentence Transformer](https://www.geeksforgeeks.org/nlp/sentence-transformer)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25/tree/master)
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)