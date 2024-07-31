import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox


# Carregar os dados (substitua 'seu_dataset.csv' pelo nome do seu arquivo de dados)
data = pd.read_csv('dados_treino2_8.csv')
data2 = pd.read_csv('dados_check2_8.csv')

# Definir as variáveis independentes (features) e a variável alvo (target)
X = data[['CAO', 'P2O5', 'FE2O3']]
y = data['DIST_MAS']

# Aplicar a transformação logarítmica (pode ser ajustada de acordo com suas necessidades)
X_transformed = np.log1p(X)

# Suponhamos que 'X' seja a matriz de dados de treinamento e 'y' seja o vetor da variável alvo
# X deve ter 10 colunas correspondentes às 10 variáveis independentes
# y deve ser a variável alvo que você deseja prever

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Criar um modelo de regressão linear
modelo_regressao = LinearRegression()

# Treinar o modelo no conjunto de treinamento
modelo_regressao.fit(X_train, y_train)

# Fazer previsões usando o modelo para o BD de processo (treino/teste) 1348
y_pred = modelo_regressao.predict(X_transformed)


# Avaliar o desempenho do modelo
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Dados de exemplo para ilustrar o gráfico
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
plt.title ('MODELO REGRESSÃO DISTMAS')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.show()

#Para verificar a normalidade dos resíduos, Histograma dos resíduos: 
# Crie um histograma dos resíduos e verifique se ele se assemelha a uma distribuição normal.

residuos = y - y_pred
plt.hist(residuos, bins=20, color='blue', alpha=0.5)
plt.title('Histograma dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()

#grafico de disperção residuos homoscedasticidade

residuos = y - y_pred
plt.scatter(y_pred, residuos, color='blue', alpha=0.5)
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos')
plt.title('Gráfico de Dispersão dos Resíduos')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Suponha que 'novos_dados' (dados de avaliação/check=350)seja uma matriz com os novos dados de entrada
# Cada linha representa um exemplo com as mesmas 9 variáveis independentes usadas para treinar o modelo
# O objetivo é prever a variável alvo para cada um desses exemplos

# Definir as variáveis independentes (features) e a variável alvo (target)
X2 = data2[['CAO', 'P2O5', 'FE2O3']]
y2 = data2['DIST_MAS']

# Aplicar a transformação logarítmica (pode ser ajustada de acordo com suas necessidades)
X_transformed2 = np.log1p(X2)

# Fazer previsões com base nos novos dados (BD avaliação)
y_pred_check = modelo_regressao.predict(X_transformed2)

# 'previsoes_novos_dados' conterá as previsões para a variável alvo com base nos novos dados
print(y_pred_check)

data2['final'] = y_pred_check

with open('saida8.csv', 'w') as arquivo:
    print((data2),file=arquivo)
data2.to_csv('saida8.csv', sep='\t', encoding='utf-8')   

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

