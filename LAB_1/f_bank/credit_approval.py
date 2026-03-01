import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

# === Загрузка данных ===
train = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/bank_scoring_train.csv", sep = "\t")
test = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/bank_scoring_test.csv", sep = "\t")

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("\nПропуски в train:")
print(train.isnull().sum())
print("\nРаспределение целевого класса:")
print(train["SeriousDlqin2yrs"].value_counts())
print(train["SeriousDlqin2yrs"].value_counts(normalize=True).round(3))

# === Препроцессинг ===
# Заполняем пропуски медианой — банковские данные, медиана безопаснее среднего
train = train.fillna(train.median(numeric_only=True))
test = test.fillna(test.median(numeric_only=True))

target = "SeriousDlqin2yrs"
X_train = train.drop(columns=[target])
y_train = train[target]
X_test = test.drop(columns=[target])
y_test = test[target]

# Скейлим для логистической регрессии
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Модели ===
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini", max_depth=6,
        min_samples_leaf=10, random_state=666
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=10, random_state=666, class_weight="balanced", n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=666
    )
}

results = {}

print("\n" + "="*65)
print(f"{'Model':>22} | {'Train AUC':>10} | {'Test AUC':>10} | {'Test Acc':>10}")
print("="*65)

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    train_auc = roc_auc_score(y_train, y_prob_train)
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob_test,
        "train_auc": train_auc,
        "test_auc": test_auc
    }

    print(f"{name:>22} | {train_auc:>10.4f} | {test_auc:>10.4f} | {test_acc:>10.4f}")

# === ROC кривые ===
# plt.figure(figsize=(8, 6))
# for name, res in results.items():
#     fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
#     plt.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.4f})")
#
# plt.plot([0, 1], [0, 1], "k--", label="Random")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curves")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# === Classification reports + Confusion matrices ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, res) in enumerate(results.items()):
    print(f"\n=== {name} ===")
    print(classification_report(y_test, res["y_pred"],
                                 target_names=["no default", "default"]))
    cm = confusion_matrix(y_test, res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["no default", "default"]).plot(
        ax=axes[i], cmap=plt.cm.Blues, colorbar=False
    )
    axes[i].set_title(name)

plt.tight_layout()
plt.show()
