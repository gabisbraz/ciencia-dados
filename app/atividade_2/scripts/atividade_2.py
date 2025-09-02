# Projeto Integrador Parte B – Preparação dos Dados

# Entregas:
# 1) Faça um relatório respondendo cada pergunta separadamente.
# 2) Link para a base utilizada.
# 3) Código completo em Python.

# Dando continuidade ao Projeto Integrador - Parte A, faça uma análise dos
# mesmos dados utilizados anteriormente, respondendo às seguintes questões:

import pandas as pd
import numpy as np

df = pd.read_csv("data/tb_1.csv")

# Sobre a limpeza de dados
# a) Há valores ausentes no conjunto de dados? Como serão tratados?
print("Valores ausentes por coluna:")
print(df.isnull().sum())

# b) Existem valores duplicados?
print("\nNúmero de linhas duplicadas:", df.duplicated().sum())
df = df.drop_duplicates()
print("Número de linhas após remover duplicatas:", len(df))

# c) Há outliers nos dados?

numeric_cols = ["math_score", "reading_score", "writing_score"]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)  
    Q3 = df[col].quantile(0.75)  
    IQR = Q3 - Q1                

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nColuna: {col}")
    print(f"Limite inferior: {lower_bound}, Limite superior: {upper_bound}")
    print(f"Número de outliers: {outliers.shape[0]}")

# Transformação e Engenharia de Atributos
# a) Há dados que precisam ser transformados de categóricos para numéricos?
df.info()
