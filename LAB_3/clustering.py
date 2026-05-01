import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv("z_datasets/pluton.csv")
X = df.values

max_iters = [1, 2, 3, 5, 10, 50, 300]
results = []

for mi in max_iters:
    km = KMeans(n_clusters=3, max_iter=mi, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    results.append({"max_iter": mi, "inertia": inertia,
                    "silhouette": sil, "davies_bouldin": db})

df_res = pd.DataFrame(results)
print("=== Влияние max_iter (без стандартизации) ===")
print(df_res.to_string(index=False))

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

km_raw = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)
km_std = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)
labels_raw = km_raw.fit_predict(X)
labels_std = km_std.fit_predict(X_std)

print("\n=== Стандартизация vs без (max_iter=300) ===")
for name, lbl, data in [("Без стандартизации", labels_raw, X),
                         ("Со стандартизацией",  labels_std, X_std)]:
    print(f"\n{name}:")
    print(f"  Inertia:        {KMeans(n_clusters=3, n_init=10, random_state=42).fit(data).inertia_:.4f}")
    print(f"  Silhouette:     {silhouette_score(data, lbl):.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin_score(data, lbl):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(df_res["max_iter"], df_res["inertia"], "o-", color="steelblue")
axes[0].set_xlabel("max_iter"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Inertia vs max_iter"); axes[0].grid(True, alpha=0.3)

axes[1].plot(df_res["max_iter"], df_res["silhouette"], "o-", color="seagreen")
axes[1].set_xlabel("max_iter"); axes[1].set_ylabel("Silhouette")
axes[1].set_title("Silhouette vs max_iter"); axes[1].grid(True, alpha=0.3)

# Scatter: raw vs std clusters (по первым двум признакам)
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for ax, lbl, title in [(axes[2], labels_std, "Кластеры (со стандартизацией)")]:
    for c in range(3):
        mask = lbl == c
        ax.scatter(X[mask, 1], X[mask, 2],
                   color=colors[c], label=f"Кластер {c+1}", alpha=0.8, edgecolors="k", lw=0.4)
    ax.set_xlabel("Pu239"); ax.set_ylabel("Pu240")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pluton_results.png", dpi=150)
plt.show()
print("\nГрафик сохранён: pluton_results.png")

df["cluster_raw"] = labels_raw
df["cluster_std"] = labels_std
print("\n=== Разбивка по кластерам (стандартизация) ===")
print(df.groupby("cluster_std").mean().round(3))
