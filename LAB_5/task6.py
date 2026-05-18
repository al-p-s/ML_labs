import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('z_datasets/JohnsonJohnson.csv')
df[['Year', 'Quarter']] = df['index'].str.extract(r'(\d+) (Q\d)')
df['Year'] = df['Year'].astype(int)
df['t'] = np.arange(len(df))

# График
fig, ax = plt.subplots(figsize=(12, 5))
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = df[df['Quarter'] == q]
    ax.plot(sub['t'], sub['value'], label=q, marker='o', ms=3)
ax.set_title('Прибыль Johnson & Johnson по кварталам')
ax.set_xlabel('Квартал')
ax.set_ylabel('Прибыль')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part6_plot.png', dpi=150)
plt.show()

# Регрессия по каждому кварталу
print("Динамика по кварталам (coef = прирост за год):")
models = {}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = df[df['Quarter'] == q].copy()
    sub['yr'] = np.arange(len(sub))
    m = LinearRegression().fit(sub[['yr']], sub['value'])
    models[q] = m
    print(f"{q}: coef={m.coef_[0]:.4f}, intercept={m.intercept_:.4f}")

# Все вместе
m_all = LinearRegression().fit(df[['t']], df['value'])
print(f"\nВсе вместе: coef={m_all.coef_[0]:.4f}")

# Прогноз на 2016
print("\nПрогноз 2016:")
last_year = df['Year'].max()
years_from_last = 2016 - last_year
total = 0
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = df[df['Quarter'] == q]
    n = len(sub) + years_from_last
    pred = models[q].predict([[n]])[0]
    print(f"{q}: {pred:.2f}")
    total += pred
print(f"Среднее за 2016: {total/4:.2f}")
