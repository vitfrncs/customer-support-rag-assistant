# Customer Support RAG Assistant

Protótipo de assistente para atendimento ao cliente que utiliza recuperação semântica e modelos de linguagem para gerar sugestões de respostas.

A aplicação recebe uma solicitação, recupera exemplos semelhantes de atendimento e utiliza essas informações para selecionar orientações que auxiliam o LLM na geração de um rascunho. A resposta pode ser revisada e editada por um atendente antes de ser enviada ao cliente.

Além da aplicação, o projeto inclui experimentos de recuperação de informação, uma comparação entre BM25 e embeddings E5, testes automatizados e uma avaliação inicial das respostas geradas.

## Funcionamento

O sistema utiliza o dataset sintético [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), que contém solicitações de atendimento, respostas associadas e rótulos de intenção.

O pipeline funciona da seguinte maneira:

1. O usuário insere uma mensagem na interface Streamlit.
2. O recuperador semântico E5 encontra as três solicitações mais semelhantes no conjunto indexado.
3. A intenção predominante entre os exemplos recuperados é utilizada para selecionar orientações específicas de atendimento.
4. As solicitações recuperadas e as orientações são organizadas em um contexto enviado ao Gemini.
5. O modelo gera um rascunho de resposta e identifica informações que precisam ser verificadas.
6. O Streamlit apresenta o rascunho e os exemplos recuperados.

```text
Mensagem do cliente
        |
        v
Recuperação semântica (E5)
        |
        v
Exemplos semelhantes
        |
        v
Seleção de orientações
        |
        v
Gemini + Pydantic
        |
        v
Rascunho para revisão humana
```

As respostas originais do Bitext também podem ser consultadas na interface, mas não são utilizadas como evidência factual na geração. Isso porque o dataset é sintético e algumas respostas descrevem ações que não foram realmente executadas.

As orientações de atendimento são fictícias e estão definidas em `data/knowledge/support_guidelines.json`. O sistema sempre utiliza as orientações gerais e acrescenta regras específicas quando identifica uma intenção configurada.


## Como executar

### 1. Preparar o ambiente

Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

### 2. Preparar os dados

Execute:

```bash
python scripts/prepare_data.py
```

O script baixa e prepara os dados do Bitext, gerando dois arquivos em `data/processed/`:

- `index_examples.csv`: exemplos utilizados na recuperação;
- `eval_queries.csv`: consultas separadas para avaliação.

Foram utilizados 25 exemplos para indexação e 5 para avaliação por intenção, totalizando 675 exemplos no índice e 135 consultas de avaliação. A divisão utiliza uma seed fixa (`42`).

### 3. Gerar os embeddings

```bash
python scripts/build_embeddings.py
```

Os embeddings são gerados com o modelo `intfloat/multilingual-e5-small` e armazenados em:

```text
data/processed/semantic_index_e5.npz
```

O modelo utiliza os prefixos `passage:` para os exemplos indexados e `query:` para as consultas. Os vetores são normalizados e comparados por similaridade de cosseno.

### 4. Configurar o Gemini

É necessário possuir uma chave de API do Gemini.

No terminal:

```bash
read -r -s -p "Gemini API key: " GEMINI_API_KEY
echo
export GEMINI_API_KEY
```

### 5. Executar a aplicação

```bash
PYTHONPATH=src python -m streamlit run app.py
```

A interface permite inserir uma solicitação, gerar um rascunho de atendimento e consultar os casos semelhantes recuperados do Bitext.

## Avaliação da recuperação

Foram avaliadas duas estratégias:

- BM25: recuperação lexical baseada na correspondência entre termos.
- E5: recuperação semântica baseada em embeddings.

Ambas utilizaram os mesmos 675 exemplos, as mesmas 135 consultas e `top_k=3`.

Um exemplo foi considerado relevante quando sua intenção correspondia à intenção da consulta.

| Métrica | BM25 | E5 |
| --- | ---: | ---: |
| Precision@3 | 0.8074 | 0.8543 |
| Recall@3 | 0.0969 | 0.1025 |
| Hit Rate@3 | 0.9556 | 0.9630 |
| MRR@3 | 0.9086 | 0.9198 |

O E5 apresentou desempenho superior nas quatro métricas gerais, embora o BM25 tenha obtido resultados melhores em algumas intenções, como `create_account`, `edit_account` e `switch_account`.

A análise dos erros mostrou dificuldades na distinção entre solicitações semanticamente próximas, especialmente `get_refund` e `check_refund_policy`.

Como existem 25 exemplos relevantes por intenção e apenas três são recuperados, o maior Recall@3 possível nessa configuração é 0,12.

Esses resultados consideram a correspondência de intenção como critério de relevância. Portanto, não representam necessariamente a relevância semântica completa de cada resultado.

Para reproduzir os experimentos:

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py
PYTHONPATH=src python scripts/evaluate_bm25.py
```

Os resultados são armazenados em `results/`, incluindo métricas por consulta, por intenção e uma comparação geral entre os métodos.

## Avaliação da geração

O pipeline de geração foi avaliado utilizando oito solicitações de atendimento, incluindo casos de rastreamento de pedidos, reembolsos, problemas de pagamento, alteração de endereço e tentativas de manipulação das instruções.

Foram realizadas duas execuções, antes e depois de ajustes nas orientações e no prompt.

A avaliação considerou aspectos como idioma, utilidade, informações necessárias e afirmações sobre ações não verificadas.

Na primeira execução, seis das oito respostas respeitaram o idioma da solicitação. Na segunda, foram quatro das oito.

Também foram observadas respostas que sugeriam capacidades inexistentes, como consultar o status de pedidos ou registrar solicitações de atendimento.

Esses resultados mostram limitações na consistência do modelo. Como a avaliação utilizou poucos casos e apenas uma execução por configuração, não é possível concluir que uma versão do prompt seja superior à outra de maneira geral.

O experimento pode ser reproduzido com:

```bash
PYTHONPATH=src python scripts/evaluate_generation.py
```

Essa avaliação utiliza a API do Gemini. O script não sobrescreve automaticamente arquivos de resultados existentes.

## Testes

Os testes automatizados utilizam Pytest e componentes simulados para verificar o comportamento do pipeline sem realizar chamadas ao Gemini.

Entre os comportamentos testados estão a validação das entradas, a geração estruturada e a separação entre exemplos recuperados e informações enviadas ao LLM.

Para executar:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Limitações

O dataset e as orientações de atendimento são sintéticos. 

A seleção de orientações específicas depende de uma heurística baseada nas intenções dos exemplos recuperados. Caso a recuperação seja incorreta ou não exista uma intenção configurada, o contexto poderá conter apenas as orientações gerais.

Além disso, as respostas do Gemini podem apresentar inconsistências de idioma ou afirmações não verificadas. As instruções do prompt e as orientações de atendimento não garantem que esses problemas sejam eliminados.

## Documentação

Os materiais de estudo e os registros dos experimentos estão organizados em `docs/learning/`.

Eles incluem explicações sobre embeddings, BM25, métricas de ranking, avaliação da recuperação e análise das respostas geradas pelo RAG.