import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import sklearn.decomposition as decomposition
import seaborn as sns

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Suponha que você tenha um DataFrame chamado 'df' com suas variáveis independentes

# Normalizar os dados antes de aplicar PCA
from sklearn.preprocessing import StandardScaler

# Separar as variáveis independentes (X)
data = pd.read_csv('temp6.csv')
X = data[['NB2O5','P2O5','SIO2','FE2O3','BAO','CAO','MGO', 'TIO2','AL2O3',
        'FE2O3C','MMAG','DISTMAS','P2O5C','LAMA','CAOC', 'MGOC','AL2O3C', 'SIO2C', 'NB2O5RF']]

# Normalizar os dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Realizar a PCA
pca = PCA(n_components=19)  # Aqui estamos usando 2 componentes principais
pca_result = pca.fit_transform(X_normalized)

# Variance explained by each component
explained_variance = pca.explained_variance_ratio_

# Extrair as cargas fatoriais
loadings = pca.components_.T

# Criar um DataFrame com as cargas fatoriais 'PC8', 'PC9','PC10','PC11', 'PC12', 'PC13', 'PC14', 'PC15','PC16', 'PC17', 'PC18','PC19'
df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6','PC7', 
                                              'PC8', 'PC9','PC10','PC11', 'PC12', 'PC13', 'PC14', 'PC15','PC16', 'PC17', 'PC18','PC19'], index=X.columns)

# Plotar as cargas fatoriais
plt.figure(figsize=(10, 8))
sns.heatmap(df_loadings, annot=True, cmap='coolwarm')
plt.title('Cargas Fatoriais da PCA')
plt.xlabel('Componentes Principais')
plt.ylabel('Variáveis')
plt.show()


# Bar plot of explained variance
plt.subplot(1, 2, 2)
plt.bar(range(1, 20), explained_variance, alpha=0.5, align='center',
        label='Variancia individual explicada')
plt.step(range(1, 20), np.cumsum(explained_variance), where='mid',
         label='Variancia explicada acumulada')
plt.title('Variancia explciada por componente')
plt.xlabel('Principal Componente')
plt.ylabel('Variância Explicada')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

















