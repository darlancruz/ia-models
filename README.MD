# Classificação com KNN, Rede Neural e Naive Bayes

Implementações em NumPy de três modelos, avaliados em dois datasets (MNIST e Breast Cancer) com o mesmo pipeline: split treino/validação/teste, normalização, seleção de hiperparâmetros na validação e avaliação final no teste.

## Como rodar

Execute sempre a partir da **raiz do projeto**.

### 1. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 2. Baixar os datasets (obrigatório antes do `main.py`)

```bash
python download_data.py
```

Isso baixa os CSVs do Google Drive e os coloca em `./data` (a pasta é criada automaticamente se não existir). Se os arquivos já estiverem lá, nada é baixado de novo. Para forçar um novo download:

```bash
python download_data.py --force
```

> **Importante:** o `main.py` apenas lê os arquivos de `./data`. Se você rodá-lo antes do `download_data.py`, vai dar erro de arquivo não encontrado.

### 3. Rodar os modelos

```bash
python main.py
```

O `main.py` carrega cada dataset e chama o `run` de cada modelo. Para cada combinação ele imprime as métricas e a tabela por classe, e salva os gráficos (matriz de confusão e curvas de desempenho) na pasta de saída configurada.

## Estrutura

```
projeto/
├── main.py              # ponto de entrada
├── requirements.txt     # dependências (pip install -r requirements.txt)
├── download_data.py     # baixa os CSVs para ./data
├── evaluation.py        # métricas, matriz de confusão, relatório (compartilhado)
├── data/                # CSVs (criada pelo download_data.py)
└── objects/
    ├── data.py          # carregamento, split e normalização (compartilhado)
    └── models/
        ├── knn.py
        ├── neural_network.py
        └── naive_bayes.py
```

## Como funciona

Os dados são preparados uma única vez por `prepare_dataset`, que devolve um objeto `Dataset` já dividido e normalizado. Cada modelo tem um método `run(dataset)` que faz o pipeline completo (a rede neural é a exceção: usa hiperparâmetros fixos por dataset, definidos em `PARAMS` no `neural_network.py`, e pula a etapa 1):

1. **Seleção de hiperparâmetros** na validação (critério: F1 macro, com a accuracy como desempate).
2. **Treino final** com os melhores hiperparâmetros em treino + validação.
3. **Avaliação no teste**, feita uma única vez.
4. **Gráficos e tabelas** salvos em arquivos `.png`.

| Modelo | Hiperparâmetros |
|---|---|
| KNN | K (ímpares de 3 a 15) e distância (euclidiana, manhattan) |
| Rede neural | **fixos** (sem seleção): MNIST = 128 neurônios, lr 0,10, 3000 épocas; Breast Cancer = 8 neurônios, lr 0,05, 500 épocas |
| Naive Bayes | `var_smoothing` |