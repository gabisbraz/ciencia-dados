# Projeto Integrador Parte B – Preparação dos Dados

# Entregas:
# 1) Faça um relatório respondendo cada pergunta separadamente.
# 2) Link para a base utilizada.
# 3) Código completo em Python.

# Dando continuidade ao Projeto Integrador - Parte A, faça uma análise dos
# mesmos dados utilizados anteriormente, respondendo às seguintes questões:

import pandas as pd

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


# Transformação e Engenharia de Atributos
# a) Há dados que precisam ser transformados de categóricos para numéricos?
df.info()
