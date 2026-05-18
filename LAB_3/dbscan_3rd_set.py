import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from itertools import product

df = pd.read_csv('z_datasets/clustering_3.csv', sep='\s+', header=None)
df.columns = ['x', 'y']
X = df.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Подбор параметров
eps_range = np.linspace(0.3, 1.0, 30)
min_samples_range = [5, 10, 15, 20]

best_score = -1
best_eps = None
best_ms = None
results = []

for eps, ms in product(eps_range, min_samples_range):
    labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
    core_mask = labels != -1
    n_clusters = len(np.unique(labels[core_mask]))
    n_noise = np.sum(labels == -1)
    if n_clusters < 2:
        continue
    score = silhouette_score(X_scaled[core_mask], labels[core_mask])
    results.append((eps, ms, n_clusters, n_noise, score))
    if score > best_score:
        best_score = score
        best_eps = eps
        best_ms = ms

print(f"Лучший результат: eps={best_eps:.3f}, min_samples={best_ms}, silhouette={best_score:.4f}")

# Топ-10
df_res = pd.DataFrame(results, columns=['eps', 'min_samples', 'n_clusters', 'noise', 'silhouette'])
print("\nТоп-10:")
print(df_res.sort_values('silhouette', ascending=False).head(10).to_string(index=False))

# Финальная визуализация
labels_best = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X_scaled)
n_noise = np.sum(labels_best == -1)

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels_best, s=10, cmap='tab10', alpha=0.8)
plt.title(f'DBSCAN (eps={best_eps:.2f}, min_samples={best_ms})\n'
          f'Кластеров: {len(np.unique(labels_best[labels_best != -1]))}, шум: {n_noise} точек')
plt.xlabel('x'); plt.ylabel('y')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan_tuned.png', dpi=150)
plt.show()
