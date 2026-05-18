import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations

df = pd.read_csv('z_datasets/reglab1.txt', sep='\t')
cols = ['z', 'x', 'y']

# Все варианты: каждая переменная как зависимая, остальные как признаки
results = []
for target in cols:
    features = [c for c in cols if c != target]
    # Полная модель (оба признака)
    X = df[features].values
    y = df[target].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    r2 = r2_score(y, y_pred)
    results.append({
        'model': f'{target} ~ {" + ".join(features)}',
        'R2': r2, 'RSS': rss,
        'coefs': dict(zip(features, model.coef_)),
        'intercept': model.intercept_
    })
    # Одиночные признаки
    for f in features:
        X1 = df[[f]].values
        m1 = LinearRegression().fit(X1, y)
        y1 = m1.predict(X1)
        rss1 = np.sum((y - y1) ** 2)
        r2_1 = r2_score(y, y1)
        results.append({
            'model': f'{target} ~ {f}',
            'R2': r2_1, 'RSS': rss1,
            'coefs': {f: m1.coef_[0]},
            'intercept': m1.intercept_
        })

res_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print(res_df[['model', 'R2', 'RSS']].to_string(index=False))

# --- Визуализация лучшей модели ---
best = res_df.iloc[0]
print(f"\nЛучшая модель: {best['model']}")
print(f"R² = {best['R2']:.4f}, RSS = {best['RSS']:.4f}")
print(f"Коэффициенты: {best['coefs']}, intercept = {best['intercept']:.4f}")

# Scatter: предсказанные vs реальные для всех моделей с двумя признаками
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, target in zip(axes, cols):
    features = [c for c in cols if c != target]
    X = df[features].values
    y = df[target].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    ax.scatter(y, y_pred, alpha=0.5, s=15)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1)
    ax.set_xlabel(f'True {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(f'{target} ~ {" + ".join(features)}\nR² = {r2:.4f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab5_part1_scatter.png', dpi=150)
plt.show()

# Residuals для лучшей модели
target = res_df.iloc[0]['model'].split(' ~ ')[0]
features = [c for c in cols if c != target]
X = df[features].values
y = df[target].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred, residuals, alpha=0.5, s=15)
axes[0].axhline(0, color='r', lw=1, ls='--')
axes[0].set_xlabel('Predicted values')
axes[0].set_ylabel('Remains')
axes[0].set_title(f'Remains — {target} ~ {" + ".join(features)}')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=25, edgecolor='k', alpha=0.7)
axes[1].set_xlabel('Remains')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Remains distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab5_part1_residuals.png', dpi=150)
plt.show()
