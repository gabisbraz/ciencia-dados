# # Projeto Integrador Parte E

# Entregas:
# 2) Implemente os algoritmos Naive Bayes, KNN e árvore de decisão para classificação.
# 3) Analise a matriz de confusão, acurácia, precisão, recall, f1_score e área sob a curva roc para todos os algoritmos.

# ### ALUNAS
# - Gabriella Braz
# - Giovana Ribeiro

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path):
    """
    Carrega e prepara os dados para classificação
    
    Args:
        data_path: caminho para o arquivo CSV
        
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, class_names
    """
    print("="*60)
    print("CARREGANDO E PREPARANDO OS DADOS")
    print("="*60)
    
    # Carregar dados
    try:
        df = pd.read_csv(data_path)
        print(f" Dados carregados com sucesso!")
        print(f"  Dimensão do dataset: {df.shape}")
    except FileNotFoundError:
        print(f" Arquivo não encontrado: {data_path}")
        print("Gerando dados simulados para demonstração...")
        # Criar dados simulados
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randint(0, 5, n_samples),
            'target': np.random.choice(['Classe_A', 'Classe_B', 'Classe_C'], n_samples)
        })
        print(f"  Dados simulados criados: {df.shape}")
    
    print(f"  Primeiras 5 linhas:")
    print(df.head())
    
    # Separar features e target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    print(f"\n Features: {list(X.columns)}")
    print(f" Target: {target_col}")
    
    # One-hot encoding para variáveis categóricas
    X = pd.get_dummies(X, drop_first=True)
    print(f" One-hot encoding aplicado. Shape final das features: {X.shape}")
    
    # Encode do target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f" Target encodado. Classes: {class_names}")
    
    # Verificar distribuição das classes
    class_counts = pd.Series(y_enc).value_counts().sort_index()
    print(f" Distribuição das classes:")
    for i, (count, class_name) in enumerate(zip(class_counts, class_names)):
        print(f"  {class_name}: {count} amostras")
    
    # Divisão treino/teste
    min_count = pd.Series(y_enc).value_counts().min()
    if min_count < 2:
        strat = None
        print(f"⚠️  Aviso: existe(m) classe(s) com apenas {min_count} amostra(s). Não será usado 'stratify'.")
    else:
        strat = y_enc
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=strat
    )
    
    print(f" Dados divididos:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f" Dados normalizados com StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names

def train_models(X_train_scaled, y_train):
    """
    Treina os três algoritmos de classificação
    
    Args:
        X_train_scaled: features de treino normalizadas
        y_train: target de treino
        
    Returns:
        dict com os modelos treinados
    """
    print(f"\n{'='*60}")
    print("TREINANDO OS ALGORITMOS")
    print("="*60)
    
    models = {}
    
    # 1. NAIVE BAYES
    print(" Treinando Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    models['Naive Bayes'] = nb_model
    print(" Naive Bayes treinado!")
    
    # 2. KNN
    print(" Treinando KNN (k=5)...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    models['KNN'] = knn_model
    print(" KNN treinado!")
    
    # 3. ÁRVORE DE DECISÃO
    print(" Treinando Árvore de Decisão...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    models['Árvore de Decisão'] = dt_model
    print(" Árvore de Decisão treinada!")
    
    print(f"\n Todos os {len(models)} modelos foram treinados com sucesso!")
    
    return models

def make_predictions(models, X_test_scaled):
    """
    Faz predições com todos os modelos
    
    Args:
        models: dicionário com os modelos treinados
        X_test_scaled: features de teste normalizadas
        
    Returns:
        predictions: dict com predições
        probabilities: dict com probabilidades
    """
    print(f"\n{'='*60}")
    print("FAZENDO PREDIÇÕES")
    print("="*60)
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f" Fazendo predições com {name}...")
        predictions[name] = model.predict(X_test_scaled)
        probabilities[name] = model.predict_proba(X_test_scaled)
        print(f" {name}: {len(predictions[name])} predições feitas")
    
    return predictions, probabilities

def evaluate_model(y_test, y_pred, y_prob, model_name):
    """
    Avalia um modelo individual e exibe os resultados
    
    Args:
        y_test: target verdadeiro
        y_pred: predições
        y_prob: probabilidades
        model_name: nome do modelo
    """
    print(f"\n===== {model_name} =====")
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("Matriz de Confusão:")
    print(cm)
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Verifica se é binário para calcular ROC
    if len(set(y_test)) == 2 and y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            print(f"Área sob a curva ROC: {roc_auc:.4f}")
        except Exception as e:
            print("Não foi possível calcular a curva ROC:", e)
    else:
        print("Curva ROC calculada separadamente para problema multiclasse.")

def calculate_all_metrics(y_test, y_pred, y_prob, class_names):
    """
    Calcula todas as métricas para um modelo
    
    Args:
        y_test: target verdadeiro
        y_pred: predições
        y_prob: probabilidades
        class_names: nomes das classes
        
    Returns:
        dict com todas as métricas
    """
    metrics = {}
    
    # Métricas básicas
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # ROC AUC
    try:
        if len(class_names) == 2:  # Binário
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
        else:  # Multiclasse
            # Binarizar para multiclasse
            y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
            metrics['roc_auc'] = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
    except:
        metrics['roc_auc'] = "N/A"
    
    return metrics

def prepare_roc_data(y_test, y_prob, class_names):
    """
    Prepara dados para curva ROC
    
    Args:
        y_test: target verdadeiro
        y_prob: probabilidades
        class_names: nomes das classes
        
    Returns:
        fpr, tpr para plotar curva ROC
    """
    if len(class_names) == 2:  # Problema binário
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        return fpr, tpr
    else:  # Problema multiclasse - usar macro average
        try:
            y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
            return fpr, tpr
        except:
            return None, None

def calculate_and_display_results(y_test, predictions, probabilities, class_names):
    """
    Calcula e exibe todos os resultados
    
    Args:
        y_test: target verdadeiro
        predictions: dict com predições
        probabilities: dict com probabilidades
        class_names: nomes das classes
        
    Returns:
        results: dict com todas as métricas
        y_probas: dict com dados para ROC
    """
    print(f"\n{'='*60}")
    print("CALCULANDO MÉTRICAS DE PERFORMANCE")
    print("="*60)
    
    # Calcular métricas para todos os modelos
    results = {}
    for name in predictions.keys():
        results[name] = calculate_all_metrics(
            y_test, predictions[name], probabilities[name], class_names
        )
    
    # Preparar dados para curvas ROC
    y_probas = {}
    for name in predictions.keys():
        y_probas[name] = prepare_roc_data(y_test, probabilities[name], class_names)
    
    print(" Métricas calculadas e dados ROC preparados!")
    
    # Exibir resultados
    print(f"\n{'='*60}")
    print("RESULTADOS DETALHADOS")
    print("="*60)
    
    for name, metrics in results.items():
        print(f"\n == {name} ==")
        print("Matriz de confusão:")
        print(metrics["confusion_matrix"])
        print(f"Acurácia: {metrics['accuracy']:.4f}")
        print(f"Precisão (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']}")
    
    return results, y_probas

def plot_roc_curves(results, y_probas):
    """
    Plota as curvas ROC
    
    Args:
        results: dict com métricas
        y_probas: dict com dados ROC
    """
    print(f"\n Plotando curvas ROC...")
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i, (name, (fpr, tpr)) in enumerate(y_probas.items()):
        if fpr is not None:
            auc_score = results[name]['roc_auc']
            if isinstance(auc_score, float):
                plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f"{name} (AUC = {auc_score:.3f})")
            else:
                plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=f"{name}")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Linha de Base (AUC = 0.5)')
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    plt.title("Curvas ROC - Comparação dos Algoritmos", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(" Curvas ROC plotadas!")

def plot_metrics_comparison(results):
    """
    Plota comparação visual das métricas
    
    Args:
        results: dict com métricas dos modelos
    """
    print(f"\n Plotando comparação das métricas...")
    
    # Preparar dados
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision_macro'] for model in models]
    recalls = [results[model]['recall_macro'] for model in models]
    f1_scores = [results[model]['f1_macro'] for model in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Criar barras
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Acurácia', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precisão', alpha=0.8, color='#ff7f0e')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8, color='#2ca02c')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8, color='#d62728')

    # Configurar gráfico
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Métricas dos Algoritmos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # Adicionar valores nas barras
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1, accuracies)
    add_value_labels(bars2, precisions)
    add_value_labels(bars3, recalls)
    add_value_labels(bars4, f1_scores)

    plt.tight_layout()
    plt.show()
    
    print(" Gráfico de comparação plotado!")

def display_final_summary(results):
    """
    Exibe resumo final dos resultados
    
    Args:
        results: dict com métricas dos modelos
    """
    print(f"\n{'='*60}")
    print("RESUMO FINAL DA ANÁLISE")
    print("="*60)
    
    # Criar DataFrame com resultados
    summary_data = []
    for model, metrics in results.items():
        summary_data.append({
            'Modelo': model,
            'Acurácia': metrics['accuracy'],
            'Precisão': metrics['precision_macro'],
            'Recall': metrics['recall_macro'],
            'F1-Score': metrics['f1_macro'],
            'ROC AUC': metrics['roc_auc'] if isinstance(metrics['roc_auc'], float) else 0.0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1-Score', ascending=False)
    
    print("\n RANKING DOS MODELOS (ordenado por F1-Score):")
    print("-" * 60)
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"{i}º lugar: {row['Modelo']}")
        print(f"   Acurácia: {row['Acurácia']:.4f}")
        print(f"   Precisão: {row['Precisão']:.4f}")
        print(f"   Recall: {row['Recall']:.4f}")
        print(f"   F1-Score: {row['F1-Score']:.4f}")
        if isinstance(row['ROC AUC'], float) and row['ROC AUC'] > 0:
            print(f"   ROC AUC: {row['ROC AUC']:.4f}")
        print()
    
    # Identificar melhor modelo
    best_model = summary_df.iloc[0]
    print(f" MELHOR MODELO GERAL: {best_model['Modelo']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    
    print(f"\n RECOMENDAÇÕES:")
    print(f"   • Para produção: Use {best_model['Modelo']} devido ao melhor F1-Score")
    print(f"   • Para interpretabilidade: Considere a Árvore de Decisão")
    print(f"   • Para velocidade: Naive Bayes é geralmente mais rápido")
    print(f"   • Para dados complexos: KNN pode capturar padrões locais")

def main():
    """Função principal que executa todo o pipeline"""
    
    print(" INICIANDO ANÁLISE DE CLASSIFICAÇÃO")
    print("Gabriella Braz & Giovana Ribeiro")
    print("="*60)
    
    # Configurar caminho dos dados
    data_path = Path("../../../data/tb_1.csv")
    
    try:
        # 1. Carregar e preparar dados
        X_train_scaled, X_test_scaled, y_train, y_test, class_names = load_and_prepare_data(data_path)
        
        # 2. Treinar modelos
        models = train_models(X_train_scaled, y_train)
        
        # 3. Fazer predições
        predictions, probabilities = make_predictions(models, X_test_scaled)
        
        # 4. Avaliar individualmente (opcional)
        print(f"\n{'='*60}")
        print("AVALIAÇÃO INDIVIDUAL DOS MODELOS")
        print("="*60)
        for name in models.keys():
            evaluate_model(y_test, predictions[name], probabilities[name], name)
        
        # 5. Calcular e exibir resultados completos
        results, y_probas = calculate_and_display_results(y_test, predictions, probabilities, class_names)
        
        # 6. Plotar visualizações
        plot_roc_curves(results, y_probas)
        plot_metrics_comparison(results)
        
        # 7. Resumo final
        display_final_summary(results)
        
        print(f"\n ANÁLISE COMPLETA FINALIZADA!")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        return None

if __name__ == "__main__":
    # Configurar matplotlib para funcionar em diferentes ambientes
    try:
        plt.style.use('default')
    except:
        pass
    
    results = main()