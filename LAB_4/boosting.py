import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загрузка
df = pd.read_csv('z_datasets/vehicle.csv')
X = df.drop(['Class'], axis=1)
y = df['Class']

# Сплит
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

# Базовые классификаторы
base_classifiers = {
    'Decision Tree': DecisionTreeClassifier(max_depth=1, random_state=666),
    'SVM': SVC(kernel='linear', probability=True, random_state=666),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=666),
    'Naive Bayes': GaussianNB()

}

# Диапазон n_estimators
n_estimators_range = range(1, 51, 5)

# Результаты
results = {name: {'train': [], 'test': []} for name in base_classifiers.keys()}

# Обучение
for name, base_clf in base_classifiers.items():
    print(f"Тестируем {name}...")
    for n_est in n_estimators_range:
        adaboost = AdaBoostClassifier(
            estimator=base_clf,
            n_estimators=n_est,
            random_state=666,

            # algorithm='SAMME'
        )
        adaboost.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, adaboost.predict(X_train))
        test_acc = accuracy_score(y_test, adaboost.predict(X_test))

        results[name]['train'].append(train_acc)
        results[name]['test'].append(test_acc)

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Зависимость точности от количества классификаторов в AdaBoost', fontsize=16)

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    ax.plot(n_estimators_range, res['train'], label='Train', marker='o', markersize=3)
    ax.plot(n_estimators_range, res['test'], label='Test', marker='s', markersize=3)
    ax.set_xlabel('Количество классификаторов')
    ax.set_ylabel('Точность')
    ax.set_title(f'{name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaboost_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Вывод лучших результатов
print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
for name, res in results.items():
    max_test_idx = np.argmax(res['test'])
    max_test_acc = res['test'][max_test_idx]
    n_est_best = list(n_estimators_range)[max_test_idx]
    print(f"\n{name}:")
    print(f"  Лучшая test точность: {max_test_acc:.4f} при n_estimators={n_est_best}")
    print(f"  Train точность: {res['train'][max_test_idx]:.4f}")