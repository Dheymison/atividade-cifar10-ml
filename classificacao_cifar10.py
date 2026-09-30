"""
Atividade - Classificacao CIFAR-10 (versao reduzida, formato ARFF)
Algoritmos avaliados: KNN, Decision Tree, Random Forest
Metricas: Acuracia e F1-score (macro)

Este script tem duas partes:
  1) Ajuste de hiperparametros (testa algumas variacoes de cada algoritmo)
  2) Avaliacao final, usando os melhores valores encontrados no ajuste

A parte 1 e opcional de rodar de novo (leva alguns minutos, principalmente
por causa da Decision Tree e do Random Forest com mais arvores). Os valores
ja encontrados estao comentados ao lado de cada teste.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

CAMINHO_ARFF = "cifar-10-small (1).arff"

RODAR_AJUSTE_DE_HIPERPARAMETROS = True  # mude para False para pular direto para a avaliacao final


def carregar_dados(caminho):
    """Le o ARFF direto com pandas (bem mais rapido que um parser generico de
    ARFF para esse arquivo, que e grande e totalmente numerico)."""
    with open(caminho) as f:
        for i, linha in enumerate(f):
            if linha.strip().upper() == "@DATA":
                linha_inicio = i + 1
                break

    colunas = [f"a{i}" for i in range(3072)] + ["class"]

    # Le ja com tipos enxutos: pixels como uint8 (0-255), classe como int8
    dtype_map = {c: "uint8" for c in colunas[:-1]}
    dtype_map["class"] = "int8"

    df = pd.read_csv(caminho, skiprows=linha_inicio, header=None,
                      names=colunas, dtype=dtype_map)
    return df


def ajuste_knn(X_train, X_test, y_train, y_test):
    print("\n--- Ajuste de hiperparametros: KNN (variando k) ---")
    melhor = None
    for k in [1, 3, 5, 7, 10, 15, 20]:
        t0 = time.time()
        modelo = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"  k={k:2d}  acuracia={acc:.4f}  f1={f1:.4f}  tempo={time.time() - t0:.1f}s")
        if melhor is None or acc > melhor[1]:
            melhor = (k, acc)
    print(f"  -> melhor k encontrado: {melhor[0]} (acuracia {melhor[1]:.4f})")
    # Resultado obtido no experimento original: melhor k = 7 (acc 0.3062),
    # praticamente igual ao k=5 padrao (acc 0.3045) -- ganho marginal, o que indica
    # que a limitacao do KNN aqui vem da alta dimensionalidade dos dados, nao do k.
    return melhor[0]


def ajuste_decision_tree(X_train, X_test, y_train, y_test):
    print("\n--- Ajuste de hiperparametros: Decision Tree (variando max_depth) ---")
    melhor = None
    for depth in [5, 10, 15, 20, None]:
        t0 = time.time()
        modelo = DecisionTreeClassifier(max_depth=depth, random_state=42)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"  max_depth={str(depth):5s}  acuracia={acc:.4f}  f1={f1:.4f}  tempo={time.time() - t0:.1f}s")
        if melhor is None or acc > melhor[1]:
            melhor = (depth, acc)
    print(f"  -> melhor max_depth encontrado: {melhor[0]} (acuracia {melhor[1]:.4f})")
    # Resultado obtido no experimento original: max_depth=10 (acc 0.2622) supera
    # a arvore sem limite (acc 0.2365) -- limitar a profundidade reduz o overfitting.
    return melhor[0]


def ajuste_random_forest(X_train, X_test, y_train, y_test):
    print("\n--- Ajuste de hiperparametros: Random Forest (variando n_estimators) ---")
    melhor = None
    for n in [100, 200]:
        t0 = time.time()
        modelo = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"  n_estimators={n:3d}  acuracia={acc:.4f}  f1={f1:.4f}  tempo={time.time() - t0:.1f}s")
        if melhor is None or acc > melhor[1]:
            melhor = (n, acc)
    print(f"  -> melhor n_estimators encontrado: {melhor[0]} (acuracia {melhor[1]:.4f})")
    # Resultado obtido no experimento original: n_estimators=200 (acc 0.4360) supera
    # o padrao de 100 arvores (acc 0.4173), ao custo de quase o dobro do tempo.
    return melhor[0]


def main():
    print("Carregando dados...")
    t0 = time.time()
    df = carregar_dados(CAMINHO_ARFF)
    print(f"Dados carregados em {time.time() - t0:.1f}s")
    print("Shape:", df.shape)
    print("\nDistribuicao das classes:")
    print(df["class"].value_counts().sort_index())
    print("\nValores nulos:", df.isna().sum().sum())

    # X e y. Pixels normalizados (0-255 -> 0-1): importante para o KNN,
    # que e sensivel a escala das variaveis (distancia entre pontos).
    X = df.drop(columns=["class"]).to_numpy(dtype=np.float32) / 255.0
    y = df["class"].to_numpy()

    # Split treino/teste mantendo a proporcao das 10 classes (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if RODAR_AJUSTE_DE_HIPERPARAMETROS:
        melhor_k = ajuste_knn(X_train, X_test, y_train, y_test)
        melhor_depth = ajuste_decision_tree(X_train, X_test, y_train, y_test)
        melhor_n = ajuste_random_forest(X_train, X_test, y_train, y_test)
    else:
        # valores encontrados no ajuste ja realizado
        melhor_k, melhor_depth, melhor_n = 7, 10, 200

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=melhor_k, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=melhor_depth, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=melhor_n, random_state=42, n_jobs=-1),
    }

    print("\nTreinando e avaliando os modelos finais (com hiperparametros ajustados)...\n")
    resultados = {}
    for nome, modelo in modelos.items():
        inicio = time.time()
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        duracao = time.time() - inicio

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        resultados[nome] = {"acuracia": acc, "f1_score": f1, "tempo_s": duracao}
        print(f"{nome:15s} - Acuracia: {acc:.4f} | F1-score (macro): {f1:.4f} | Tempo: {duracao:.1f}s")

    return resultados


if __name__ == "__main__":
    main()
