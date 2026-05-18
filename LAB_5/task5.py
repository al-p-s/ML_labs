import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('z_datasets/eustock.csv')
df['t'] = np.arange(len(df))

fig, ax = plt.subplots(figsize=(10, 5))
for col in ['DAX', 'SMI', 'CAC', 'FTSE']:
    ax.plot(df['t'], df[col], label=col)
ax.set_title('Котировки бирж')
ax.set_xlabel('День')
ax.set_ylabel('Котировка')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part5_plot.png', dpi=150)
plt.show()

# Регрессия для каждой биржи отдельно
print("Отдельные модели (coef = динамика в день):")
for col in ['DAX', 'SMI', 'CAC', 'FTSE']:
    m = LinearRegression().fit(df[['t']], df[col])
    print(f"{col}: coef={m.coef_[0]:.4f}")

# Все вместе — стек всех бирж
X_all = np.tile(df[['t']].values, (4, 1))
y_all = np.concatenate([df[col].values for col in ['DAX', 'SMI', 'CAC', 'FTSE']])
m_all = LinearRegression().fit(X_all, y_all)
print(f"\nВсе вместе: coef={m_all.coef_[0]:.4f}")
