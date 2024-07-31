import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Carregar os dados (substitua 'seu_dataset.csv' pelo nome do seu arquivo de dados)
data = pd.read_csv('dados7.csv')
data2= pd.read_csv ('dados_check7.csv')
###data2 = pd.read_csv('dados_check2_2.csv'), não consegui fazr deu erro

# Suponha que você tenha um DataFrame chamado 'df' com suas 10 variáveis independentes e a variável alvo 'y'
#'NB2O5', 'P2O5', 'SIO2', 'FE2O3', 'BAO', 'CAO', 'MGO', 'TIO2', 'AL2O3'
X = data[['NB2O5', 'P2O5', 'SIO2', 'FE2O3', 'BAO', 'CAO', 'MGO', 'TIO2', 'AL2O3']]
y = data['MMAG']

# Normalizar os dados para o intervalo [0, 1]
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Dividir os dados normalizados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Criar um modelo de Support Vector Machine para regressão (SVR)
modelo_svm = SVR(kernel='rbf', C=100, epsilon= 0.001)  # Você pode escolher o kernel desejado, como 'linear', 'rbf', 'poly', etc.

# Treinar o modelo no conjunto de treinamento
modelo_svm.fit(X_train, y_train)

# Fazer previsões no conjunto BD processo
y_pred = modelo_svm.predict(X_normalized)

# Fazer previsões no conjunto de teste
y_pred2 = modelo_svm.predict(X_test)

# Avaliar o desempenho do modelo
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Avaliar o desempenho do modelo usando a métrica de erro médio quadrático (MSE)
mse = mean_squared_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

#Dados de exemplo para ilustrar o gráfico
valores_reais = np.array([y])
valores_preditos = np.array([y_pred])

# Calcular o coeficiente de correlação de Pearson (R-squared)
coeficiente_correlacao = np.corrcoef(y, y_pred)[0, 1]
coeficiente_r2 = r2_score(y, y_pred)

# Criar um gráfico de dispersão (scatter plot)
plt.scatter(valores_reais, valores_preditos, c='blue', label=f'Coef. de Correlação: {coeficiente_correlacao:.2f}\nR-squared: {coeficiente_r2:.2f}')

# Configurar rótulos dos eixos
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title ('SVM DIST_MAS')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.show()
















