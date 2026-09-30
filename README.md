# Atividade CIFAR-10 - Aprendizagem de Máquina

Comparação dos algoritmos KNN, Decision Tree e Random Forest na classificação do dataset CIFAR-10 (versão reduzida), avaliados por acurácia e F1-score.

## Dataset

Este repositório não inclui o arquivo `.arff` (209 MB, acima do limite do GitHub). Baixe o dataset **cifar-10-small** no OpenML:

https://www.openml.org/search?type=data&search=cifar-10-small

Coloque o arquivo baixado na mesma pasta do script, com o nome `cifar-10-small (1).arff` (ou ajuste o nome no início do script).

## Como rodar

```
pip install -r requirements.txt
python classificacao_cifar10.py
```

## Conteúdo

- `classificacao_cifar10.py`: carrega os dados, ajusta hiperparâmetros (k do KNN, profundidade da árvore, número de árvores da RF) e avalia os três modelos.
- `requirements.txt`: bibliotecas necessárias.
