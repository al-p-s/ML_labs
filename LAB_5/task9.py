import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('z_datasets/nsw74psid1.csv')
X = df.drop(columns='re78').values
y = df['re78'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)

# SVR (нужна нормализация)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
svr = SVR(kernel='rbf', C=1).fit(X_train_s, y_train)

models = {
    'Decision Tree': (dt, X_train, X_test),
    'Linear Regression': (lr, X_train, X_test),
    'SVR': (svr, X_train_s, X_test_s)
}

print(f"{'Модель':<20} {'Train MSE':>12} {'Test MSE':>12} {'Test R²':>10}")
for name, (m, Xtr, Xte) in models.items():
    tr_mse = mean_squared_error(y_train, m.predict(Xtr))
    te_mse = mean_squared_error(y_test, m.predict(Xte))
    te_r2 = r2_score(y_test, m.predict(Xte))
    print(f"{name:<20} {tr_mse:>12.1f} {te_mse:>12.1f} {te_r2:>10.4f}")

# График
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, (m, Xtr, Xte)) in zip(axes, models.items()):
    y_pred = m.predict(Xte)
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
    ax.set_title(name)
    ax.set_xlabel('Реальные re78')
    ax.set_ylabel('Предсказанные re78')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part9.png', dpi=150)
plt.show()
