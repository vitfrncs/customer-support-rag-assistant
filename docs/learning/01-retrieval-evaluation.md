# Avaliando a recuperação semântica

Queria entender melhor se o E5 está recuperando exemplos que realmente têm relação com a mensagem do cliente. Até agora eu tinha testado algumas consultas manualmente e os resultados pareciam fazer sentido, mas isso não diz muito sobre o desempenho do recuperador no geral.

Por enquanto, estou avaliando só a recuperação, não as respostas geradas pelo Gemini.

Usei as solicitações do Bitext. Depois de normalizar os textos e remover instruções duplicadas, separei 25 exemplos de cada intenção para o índice e outros 5 para avaliação (seed 42). Como o dataset tem 27 intenções, fiquei com 675 exemplos no índice e 135 consultas de teste.

O modelo é o `intfloat/multilingual-e5-small`. Ele gera embeddings para as solicitações do índice usando `passage:` e para as consultas usando `query:`. Os vetores são normalizados, então consigo calcular a similaridade por cosseno usando o produto escalar. Para cada consulta, recupero os três exemplos com maior similaridade, que é o mesmo `top_k=3` usado no app.py.

Para medir os resultados, considerei que um exemplo é relevante quando tem a mesma `intent` da consulta. É uma maneira simples de avaliar automaticamente, mas não é uma definição perfeita de relevância, já que duas solicitações podem ter intenções diferentes e ainda assim ser parecidas, ou ter a mesma intenção e tratar de problemas específicos diferentes.

## As métricas

Todas as métricas usam os três primeiros resultados (`@3`). 

Um exemplo para entender a diferença entre elas: Imagine uma consulta `track_order` que retorna, nesta ordem:

```text
1. track_order
2. delivery_period
3. track_order
```

Precision@3 mede quantos resultados recuperados são relevantes. Nesse exemplo, dois dos três têm a intenção correta:

$$
P@3 = \frac{2}{3} \approx 0{,}667
$$

Recall@3 mede quantos dos exemplos relevantes existentes no índice foram encontrados. Como tenho 25 exemplos de cada intenção no índice, nesse caso seria:

$$
R@3 = \frac{2}{25} = 0{,}08
$$

Uma coisa importante: com `top_k=3`, o maior Recall@3 possível nesse experimento é `3/25 = 0,12`. Então um recall de 0,10 não significa que o recuperador foi necessariamente ruim. É uma consequência de recuperar só três exemplos entre os 25 relevantes disponíveis.

Hit Rate@3 é mais simples: verifica se apareceu pelo menos um resultado relevante nos três primeiros. Para cada consulta, vale 1 quando existe algum acerto e 0 quando não existe. No exemplo acima, o hit é 1. A média entre todas as consultas mostra a proporção de consultas em que o recuperador conseguiu encontrar pelo menos um exemplo da intenção esperada. Essa métrica não diferencia um acerto de três acertos.

MRR@3 (*Mean Reciprocal Rank*) leva em conta a posição do primeiro resultado relevante. Se ele aparece na primeira posição, o valor é 1; na segunda, 1/2; na terceira, 1/3. Se não aparece no top-3, o valor é 0. Depois faço a média desses valores entre todas as consultas. No exemplo, o valor é 1, porque o primeiro resultado já é `track_order`.

Ou seja: Precision considera a quantidade de acertos no top-3, Hit Rate verifica se houve pelo menos um acerto e MRR considera a posição do primeiro acerto. Recall considera quantos dos relevantes disponíveis foram recuperados.

## Resultados do E5

Rodei a avaliação nas 135 consultas e calculei a média das métricas:

```text
Precision@3    0.8543
Recall@3       0.1025
Hit Rate@3     0.9630
MRR@3          0.9198
```

Dos 405 resultados recuperados no total (135 consultas × 3), 346 tinham a mesma intenção da consulta. Em 130 das 135 consultas apareceu pelo menos um exemplo da intenção correta. Além disso, em 119 consultas o primeiro resultado já era relevante, e em 94 consultas os três resultados tinham a intenção esperada.

O E5 foi especialmente consistente em intenções como `track_order`, `delivery_period`, `payment_issue` e `contact_human_agent`: nas cinco consultas avaliadas para cada uma, os três resultados recuperados tinham a intenção correta.

Já `get_refund` foi a intenção com menor Precision@3 (0,467). Em duas das cinco consultas de reembolso, nenhum dos três exemplos tinha a intenção esperada. Isso me chamou a atenção porque o pipeline usa as intenções recuperadas para escolher orientações de atendimento.

Fui ver as cinco consultas que tiveram Hit Rate@3 igual a zero. Algumas confusões fazem bastante sentido olhando os textos:

- `i do not know what to do to cdheck my bill #85632` (`check_invoice`) recuperou três exemplos de `get_invoice`. Verificar uma fatura e obtê-la não são exatamente a mesma coisa, mas são solicitações muito próximas.
- `want assistance ot close my standard account` (`delete_account`) recuperou dois exemplos de `switch_account` e um de `recover_password`. Aqui o erro é mais sério, porque fechar uma conta é bem diferente de trocar ou recuperar o acesso a ela.
- `I want help asking for an reimbursement` (`get_refund`) recuperou três exemplos de `check_refund_policy`. O modelo confundiu pedir reembolso com perguntar em quais situações ele é permitido.
- `I don't know how to get refunds` (`get_refund`) recuperou dois exemplos de `check_refund_policy` e um de `track_refund`. A própria consulta é um pouco ambígua, então a confusão é compreensível.
- `I want assistance to use the premium account` (`switch_account`) recuperou dois exemplos de `create_account` e um de `edit_account`. Também é uma mensagem pouco específica: não fica tão claro se a pessoa quer criar, mudar ou usar uma conta.

No primeiro caso de reembolso, a similaridade do resultado mais próximo foi 0,9315, apesar de a intenção ser diferente. Isso ajuda a lembrar que similaridade alta não é probabilidade de a intenção estar correta. O E5 encontra textos parecidos, mas nem sempre diferencia bem a ação que a pessoa quer realizar.

Esse problema importa para o RAG porque escolho as orientações específicas a partir da intenção predominante nos exemplos recuperados. Se uma consulta `get_refund` retornar apenas `check_refund_policy`, talvez as regras específicas para pedidos de reembolso não sejam selecionadas. Por isso, regras importantes, como não confirmar a aprovação de um reembolso sem verificação, não deveriam depender apenas dessa heurística.

No geral, achei o resultado bom para essa primeira versão, mas ainda não dá para dizer que o E5 é melhor que outras abordagens. Também vale lembrar que o Bitext é sintético, tem muitas frases parecidas e que usei só cinco consultas de avaliação por intenção. Além disso, essas métricas não dizem se o Gemini produz respostas corretas ou seguras.

O próximo passo é usar o BM25 como baseline, com os mesmos 675 exemplos, as mesmas 135 consultas e o mesmo top-3. Quero ver principalmente se ele distingue melhor verbos como `close`, `check` e `get`, ou se perde desempenho quando a consulta tem erros de escrita.

Com os dados preparados e o índice E5 já gerado, executei na raiz do projeto:

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py
```

O script salva os resultados individuais em `results/retrieval_e5_queries.csv` e as médias por intenção em `results/retrieval_e5_by_intent.csv`.