# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.cluster import KMeans # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

colors_list = ['red', 'blue', 'green', 'orange', 'magenta', 'deepskyblue', 'chartreuse', 'gold', 'brown',
               'blueviolet', 'darkblue', 'darkgreen', 'goldenrod', 'salmon', 'steelblue']

c1_a1 = np.random.normal(48, 2, 100) # mean and standard deviation
c1_a2 = np.random.normal(48, 2, 100)

c2_a1 = np.random.normal(36, 2, 100)
c2_a2 = np.random.normal(36, 2, 100)

c3_a1 = np.random.normal(40, 2, 100)
c3_a2 = np.random.normal(40, 2, 100)

data_a1 = np.concatenate([c1_a1, c2_a1, c3_a1])
data_a2 = np.concatenate([c1_a2, c2_a2, c3_a2])

data1 = data_a1[np.newaxis].T
data2 = data_a2[np.newaxis].T

data = np.hstack([data1, data2])

#plt.scatter(data[:, 0], data[:, 1])

km = KMeans(n_clusters=4)
km.fit(data)

labels = km.labels_
sse = km.inertia_

#j = [3819.89, 2394.69, 1764.05, 1597.59]

print(round(sse, 2))

plt.scatter(data[:, 0], data[:, 1], c=[colors_list[i] for i in labels])

""" Exemplo com dados fictícios da busca de um joelho/cotovelo ao variar k e observar valores de J. O joelho/cotovelo indica o valor ótimo de k!"""

k = [2, 3, 4, 5, 6, 7, 8]
ssq = [56, 48, 36, 35, 33, 32, 31]

plt.plot(k, ssq)

colors_list = ['red', 'blue', 'green', 'orange', 'magenta', 'deepskyblue', 'chartreuse', 'gold', 'brown',
               'blueviolet', 'darkblue', 'darkgreen', 'goldenrod', 'salmon', 'steelblue']

data_udist = np.zeros([750, 2])
data_udist[:, 0] = np.random.uniform(0, 120, 750)
data_udist[:, 1]  = np.random.uniform(0, 160, 750)

#plt.scatter(data_udist[:, feat1], data_udist[:, feat2])

km = KMeans(n_clusters=15)
km.fit(data_udist)

labels = km.labels_
sse = km.inertia_

print(round(sse, 2))

plt.scatter(data_udist[:, 0], data_udist[:, 1], c=[colors_list[i] for i in labels])

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

colors_list = ['red', 'blue', 'green', 'orange', 'magenta', 'deepskyblue', 'chartreuse', 'gold', 'brown',
               'blueviolet', 'darkblue', 'darkgreen', 'goldenrod', 'salmon', 'steelblue']

data_path = 'https://raw.githubusercontent.com/luizfsc/datasets/master/ruspini.csv'
df_data = pd.read_csv(data_path) #, header=None)
data = df_data.values
feat1, feat2 = 0, 1

km = KMeans(n_clusters=4)
km.fit(data)
labels = km.labels_

#plt.scatter(data[:, feat1], data[:, feat2])

print(round(km.inertia_,2))

# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
silhouette_avg = silhouette_score(data, labels)

print(silhouette_avg)

plt.scatter(data[:, feat1], data[:, feat2], c=[colors_list[i] for i in labels])

"""Com base nos códigos apresentados acima desenvolva um código capaz de verificar um joelho/cotovelo variando k = {2, 3, ..., 10} no k-Médias atuando sobre a base de dados Ruspini (mostre um gráfico como feito anteriormente)."""

listaK = []
ssq = []
silhouette = []

for numK in range(2, 10, 1):
  km = KMeans(n_clusters = numK)
  km.fit(data)
  #Adicionando valores a lista de variações de k
  listaK.append(numK)
  #Adicionando valores de J para cada k
  ssq.append(round(km.inertia_))
  #Adicionando os valores de silhouette para cada k
  silhouette.append(silhouette_score(data, km.labels_))
  

#Plotando o grafico
x = np.array(listaK)
y = np.array(ssq)
plt.plot(x, y)
#encontrando o melhor valor de silhouette
maxSilhouette = max(silhouette)
indexMaxSilhouette = silhouette.index(maxSilhouette)
BestKSilhouette = listaK[indexMaxSilhouette]
#mostrando qual o melhor valor de silhouette 
print(BestKSilhouette)