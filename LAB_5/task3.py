import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('z_datasets/cygage.txt', sep='\t')
X = df[['Depth']].values
y = df['calAge'].values
w = df['Weight'].values

# Без весов
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# С весами
model_w = LinearRegression().fit(X, y, sample_weight=w)
y_pred_w = model_w.predict(X)

print(f"Без весов:  R²={r2_score(y, y_pred):.4f}, RSS={np.sum((y-y_pred)**2):.1f}")
print(f"С весами:   R²={r2_score(y, y_pred_w):.4f}, RSS={np.sum((y-y_pred_w)**2):.1f}")
print(f"\nБез весов:  coef={model.coef_[0]:.4f}, intercept={model.intercept_:.2f}")
print(f"С весами:   coef={model_w.coef_[0]:.4f}, intercept={model_w.intercept_:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, s=w*100, alpha=0.7, label='Данные (размер = вес)')
plt.plot(X, y_pred, 'r--', label=f'Без весов R²={r2_score(y, y_pred):.4f}')
plt.plot(X, y_pred_w, 'g-', label=f'С весами R²={r2_score(y, y_pred_w):.4f}')
plt.xlabel('Depth')
plt.ylabel('calAge')
plt.title('Регрессия возраста от глубины')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part3.png', dpi=150)
plt.show()
