import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('z_datasets/longley.csv')
df = df.drop(columns='Population')

X = df.drop(columns='Employed').values
y = df['Employed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=666)

# Линейная регрессия
lr = LinearRegression().fit(X_train, y_train)
lr_train_err = mean_squared_error(y_train, lr.predict(X_train))
lr_test_err = mean_squared_error(y_test, lr.predict(X_test))

# Гребневая регрессия
lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_train_errs = []
ridge_test_errs = []
for lam in lambdas:
    ridge = Ridge(alpha=lam).fit(X_train, y_train)
    ridge_train_errs.append(mean_squared_error(y_train, ridge.predict(X_train)))
    ridge_test_errs.append(mean_squared_error(y_test, ridge.predict(X_test)))

print(f"Linear — train MSE: {lr_train_err:.4f}, test MSE: {lr_test_err:.4f}")
for lam, tr, te in zip(lambdas, ridge_train_errs, ridge_test_errs):
    print(f"Ridge λ={lam:<6} — train MSE: {tr:.4f}, test MSE: {te:.4f}")

plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, ridge_train_errs, 'b-o', label='Ridge train')
plt.semilogx(lambdas, ridge_test_errs, 'r-o', label='Ridge test')
plt.axhline(lr_train_err, color='b', ls='--', label='LR train')
plt.axhline(lr_test_err, color='r', ls='--', label='LR test')
plt.xlabel('λ')
plt.ylabel('MSE')
plt.title('Линейная vs Гребневая регрессия')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part4.png', dpi=150)
plt.show()
