import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Carregar o dataset Iris
df = pd.read_csv("app/atividade_4/data/diabetes.csv")

df.dtypes

# Separar variáveis preditoras e alvo
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Criar e treinar o classificador de árvore de decisão
decision_T = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=25,
    min_samples_leaf=8,
    random_state=7,
)
decision_T = decision_T.fit(X_train, y_train)

# Visualizar a árvore
plt.figure(figsize=(100, 90))
plot_tree(
    decision_T,
    filled=True,
    feature_names=X.columns,
    class_names=df["Outcome"].astype(str).unique(),
)
plt.show()

# Fazer previsões
y_pred = decision_T.predict(X_test)

# Avaliar o modelo
print("Acurácia:", metrics.accuracy_score(y_test, y_pred))

# Importância das variáveis
importances = decision_T.feature_importances_
for i, feature in enumerate(X.columns):
    print(f"{feature}: {importances[i]:.4f}")
