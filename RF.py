# Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox


# Carregar os dados (substitua 'seu_dataset.csv' pelo nome do seu arquivo de dados)
data = pd.read_csv('dados_treino2_8.csv')
data2 = pd.read_csv('dados_check2_8.csv')

# Definir as variáveis independentes (features) e a variável alvo (target)
X = data[['P2O5','CAO', 'FE2O3']]
y = data['DIST_MAS']

# Aplicar a transformação logarítmica (pode ser ajustada de acordo com suas necessidades)
X_transformed = np.log1p(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


# Criar um modelo de Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42)

# Treinar o modelo no conjunto de treinamento
random_forest.fit(X_train, y_train)

# Calcular a importância das variáveis
importancias_variaveis = random_forest.feature_importances_

# As importâncias das variáveis serão armazenadas na variável 'importancias_variaveis'
print(importancias_variaveis)

# Fazer previsões no conjunto BD processo
y_pred = random_forest.predict(X_transformed)

# Avaliar o desempenho do modelo
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

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
plt.title ('RF DISTMAS')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.show()

# Suponha que 'novos_dados' (dados de avaliação=350)seja uma matriz com os novos dados de entrada
# Cada linha representa um exemplo com as mesmas 9 variáveis independentes usadas para treinar o modelo
# O objetivo é prever a variável alvo para cada um desses exemplos

# Definir as variáveis independentes (features) e a variável alvo (target)
X2 = data2[['P2O5','CAO', 'FE2O3']]
y2 = data2['DIST_MAS']

# Aplicar a transformação logarítmica (pode ser ajustada de acordo com suas necessidades)
X_transformed2 = np.log1p(X2)

# Fazer previsões com base nos novos dados (BD avaliação)
y_pred_check = random_forest.predict(X_transformed2)

# 'previsoes_novos_dados' conterá as previsões para a variável alvo com base nos novos dados
print(y_pred_check)

data2['final'] = y_pred_check

with open('saidaCBMG.csv', 'w') as arquivo:
    print((data2),file=arquivo)
data2.to_csv('saidaCBMG.csv', sep='\t', encoding='utf-8')   

# Dados de exemplo para ilustrar o gráfico (substitua pelo seus dados reais)
valores_reais = np.array([y2])
valores_preditos = np.array([y_pred_check])

# Calcular o coeficiente de correlação de Pearson (R-squared)
coeficiente_correlacao = np.corrcoef(y2, y_pred_check)[0, 1]
coeficiente_r2 = r2_score(y2, y_pred_check)

# Criar um gráfico de dispersão (scatter plot)
plt.scatter(valores_reais, valores_preditos, c='blue', label=f'Coef. de Correlação: {coeficiente_correlacao:.2f}\nR-squared: {coeficiente_r2:.2f}')

# Configurar rótulos dos eixos
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title ('DISTMAS_check')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.show()