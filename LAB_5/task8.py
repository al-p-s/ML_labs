import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

df = pd.read_csv('z_datasets/svmdata6.txt', sep='\t', usecols=[1, 2], header=0, names=['X', 'Y'])
X = df[['X']].values
y = df['Y'].values

epsilons = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
mses = []
for eps in epsilons:
    svr = SVR(kernel='rbf', C=1, epsilon=eps).fit(X, y)
    mses.append(mean_squared_error(y, svr.predict(X)))
    print(f"ε={eps}: MSE={mses[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(epsilons, mses, 'b-o')
plt.xlabel('ε')
plt.ylabel('MSE')
plt.title('MSE от ε (SVR, C=1, rbf)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_part8.png', dpi=150)
plt.show()
