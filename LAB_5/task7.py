import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('z_datasets/cars.csv')
X = df[['speed']].values
y = df['dist'].values

# Линейная
lr = LinearRegression().fit(X, y)

# Полиномиальная 2й степени
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
pr = LinearRegression().fit(X_poly, y)

x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, label='Данные')
plt.plot(x_range, lr.predict(x_range), 'r-', label=f'Линейная')
plt.plot(x_range, pr.predict(poly.transform(x_range)), 'g-', label=f'Полиномиальная (deg=2)')
plt.xlabel('Скорость (mph)')
plt.ylabel('Тормозной путь (ft)')
plt.title('Зависимость тормозного пути от скорости')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part7.png', dpi=150)
plt.show()

from sklearn.metrics import r2_score
print(f"Линейная R²={r2_score(y, lr.predict(X)):.4f}")
print(f"Полиномиальная R²={r2_score(y, pr.predict(X_poly)):.4f}")

pred_speed = np.array([[40]])
print(f"\nПрогноз при 40 mph:")
print(f"Линейная: {lr.predict(pred_speed)[0]:.1f} ft")
print(f"Полиномиальная: {pr.predict(poly.transform(pred_speed))[0]:.1f} ft")
