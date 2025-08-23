# Projeto Integrador – Análise Exploratória de Dados
# Entregas:
# 1) Faça um relatório respondendo cada pergunta separadamente.
# 2) Link para a base utilizada.
# 3) Código completo em Python.

# 1) Escolha do Tema
# A base de dados utilizada reúne informações de estudantes do ensino médio,
# incluindo características demográficas e socioeconômicas, como gênero,
# grupo étnico, nível de escolaridade dos pais, tipo de almoço e
# participação em curso preparatório, além das notas obtidas em matemática,
# leitura e escrita (variáveis numéricas de 0 a 100).

# 2) Escolha da Base
# https://www.kaggle.com/datasets/pankajsingh016/student-performance-dataset?resource=download


# 3) Compreensão dos dados
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


df = pd.read_csv("data/atividade_1/tb_1.csv")


# a) Qual é a estrutura do dataset? (Quantas linhas e colunas existem?)
linhas = df.shape[0]
colunas = df.shape[1]
print(f"O dataset possui {linhas} linhas e {colunas} colunas ({df.columns}).")


# b) Quais são os tipos de variáveis presentes? (Numéricas, categóricas, texto, etc.)
tipos_variaveis = df.dtypes.value_counts()
print("Tipos de variáveis presentes no dataset:")
print(tipos_variaveis)


# c) Existe algum problema de valores ausentes no dataset? Como lidar com eles?
valores_ausentes = df.isnull().sum()
print("Valores ausentes por coluna:")
print(valores_ausentes)
# Se houver valores ausentes, podemos optar por removê-los ou preenchê-los.


# d) Existem valores duplicados ou inconsistências nos dados?
valores_duplicados = df.duplicated().sum()
print(f"Número de valores duplicados no dataset: {valores_duplicados}")

# 4) Estatísticas Descritivas

# a) Quais são as estatísticas básicas das variáveis numéricas? (Média, mediana, mínimo, máximo, desvio padrão).
df_calculado_estatistica = (
    df[["math_score", "reading_score", "writing_score"]]
    .agg(["mean", "median", "min", "max", "std"])
    .T
)

df_calculado_estatistica.rename(
    columns={
        "mean": "Média",
        "median": "Mediana",
        "min": "Mínimo",
        "max": "Máximo",
        "std": "Desvio Padrão",
    },
    inplace=True,
)

print(df_calculado_estatistica)

# 5) Visualização e Padrões

# a) Existe uma relação entre variáveis numéricas? (Correlação entre idade e salário, por exemplo).
df_num = df[["math_score", "reading_score", "writing_score"]]
corr_matrix = df_num.corr()
print("Matriz de correlação entre variáveis numéricas:")
print(corr_matrix)


# 1) Histograma
fig = px.histogram(
    df, x="math_score", nbins=20, title="Distribuição das notas de Matemática"
)
fig.show()


# 2) Boxplot por categoria
fig = px.box(df, x="gender", y="math_score", title="Notas de Matemática por Gênero")
fig.show()


# 3) Scatterplot (relação entre leitura e escrita)
fig = px.scatter(
    df,
    x="reading_score",
    y="writing_score",
    color="gender",
    title="Relação entre Leitura e Escrita (por gênero)",
)
fig.show()


# 4) Heatmap de correlação
corr = df[["math_score", "reading_score", "writing_score"]].corr()

fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=corr.round(2).values,
    colorscale="RdBu",
    showscale=True,
    reversescale=True,
)
fig.update_layout(title="Correlação entre variáveis numéricas")
fig.show()
