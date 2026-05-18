import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression

df = pd.read_csv('z_datasets/reglab.txt', sep='\t')
y = df['y'].values
X = df.drop(columns='y')
feature_names = X.columns.tolist()

results = []
for k in range(1, len(feature_names)):  # 1 до 3 признаков (не все 4)
    best_rss = np.inf
    best_subset = None
    for subset in combinations(feature_names, k):
        Xk = X[list(subset)].values
        model = LinearRegression().fit(Xk, y)
        rss = np.sum((y - model.predict(Xk)) ** 2)
        if rss < best_rss:
            best_rss = rss
            best_subset = subset
    results.append({'k': k, 'features': best_subset, 'RSS': best_rss})
    print(f"k={k}: {best_subset}, RSS={best_rss:.4f}")

# Лучшее подмножество по минимуму RSS
best = min(results, key=lambda r: r['RSS'])
print(f"\nОптимальное: k={best['k']}, признаки={best['features']}, RSS={best['RSS']:.4f}")
