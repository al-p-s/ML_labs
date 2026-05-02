import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------
# Загрузка данных
# ----------------------------------------------
files = {
    "z_datasets/clustering_1.csv": "clustering_1",
    "z_datasets/clustering_2.csv": "clustering_2",
    "z_datasets/clustering_3.csv": "clustering_3"
}

# Для накопления метрик
metrics_summary = []

for file, name in files.items():
    print(f"\n{'='*60}")
    print(f"Обработка: {file}")
    df = pd.read_csv(file, sep='\s+', header=None)
    df.columns = ['x', 'y']
    X = df.values
    print(f"Размер данных: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = range(2, 11)
    inertias = []
    sil_kmeans = []
    sil_agg = []
    for k in ks:
        # KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_km = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_kmeans.append(silhouette_score(X_scaled, labels_km))

        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_agg = agg.fit_predict(X_scaled)
        sil_agg.append(silhouette_score(X_scaled, labels_agg))

    # График локтя и силуэта для KMeans
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertias, 'o-', color='steelblue')
    ax1.set_xlabel('Число кластеров k')
    ax1.set_ylabel('Inertia')
    ax1.set_title(f'Локтевой метод (KMeans) – {name}')
    ax1.grid(alpha=0.3)

    ax2.plot(ks, sil_kmeans, 'o-', color='seagreen', label='KMeans')
    ax2.plot(ks, sil_agg, 's--', color='darkorange', label='Agglomerative')
    ax2.set_xlabel('Число кластеров k')
    ax2.set_ylabel('Silhouette')
    ax2.set_title(f'Коэффициент силуэта – {name}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{name}_optimal_k.png', dpi=150)
    plt.show()

    # Выбираем оптимальное k по максимуму силуэта (можно локтевой метод)
    best_k_kmeans = ks[np.argmax(sil_kmeans)]
    best_k_agg = ks[np.argmax(sil_agg)]
    print(f"Оптимальное k для KMeans: {best_k_kmeans} (silhouette={max(sil_kmeans):.4f})")
    print(f"Оптимальное k для Agglomerative: {best_k_agg} (silhouette={max(sil_agg):.4f})")

    # ------------------------------------------
    # 2. Подбор параметров DBSCAN
    # ------------------------------------------
    # Сетка параметров
    eps_range = np.linspace(0.1, 1.5, 20)
    min_samples_range = [3, 5, 7, 10]
    best_score = -1
    best_eps = None
    best_ms = None
    for eps, ms in product(eps_range, min_samples_range):
        dbs = DBSCAN(eps=eps, min_samples=ms)
        labels = dbs.fit_predict(X_scaled)
        # Исключаем шум (-1)
        core_mask = labels != -1
        if len(np.unique(labels[core_mask])) < 2:
            continue  # нужно минимум 2 кластера для силуэта
        try:
            score = silhouette_score(X_scaled[core_mask], labels[core_mask])
        except:
            score = -2
        if score > best_score:
            best_score = score
            best_eps = eps
            best_ms = ms

    print(f"Лучшие параметры DBSCAN: eps={best_eps:.3f}, min_samples={best_ms} (silhouette={best_score:.4f})")

    # ------------------------------------------
    # 3. Финальная кластеризация тремя методами
    # ------------------------------------------
    # KMeans
    km_final = KMeans(n_clusters=best_k_kmeans, n_init=10, random_state=42)
    labels_km = km_final.fit_predict(X_scaled)

    # Agglomerative
    agg_final = AgglomerativeClustering(n_clusters=best_k_agg, linkage='ward')
    labels_agg = agg_final.fit_predict(X_scaled)

    # DBSCAN
    db_final = DBSCAN(eps=best_eps, min_samples=best_ms)
    labels_db = db_final.fit_predict(X_scaled)

    # ------------------------------------------
    # 4. Метрики
    # ------------------------------------------
    # Для DBSCAN оцениваем только на нешумовых точках
    core_mask = labels_db != -1
    X_db_core = X_scaled[core_mask]
    labels_db_core = labels_db[core_mask]

    metrics = {
        'Датасет': name,
        'Метод': 'KMeans',
        'Silhouette': silhouette_score(X_scaled, labels_km),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, labels_km),
        'Calinski-Harabasz': calinski_harabasz_score(X_scaled, labels_km)
    }
    metrics_summary.append(metrics)

    metrics = {
        'Датасет': name,
        'Метод': 'Agglomerative',
        'Silhouette': silhouette_score(X_scaled, labels_agg),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, labels_agg),
        'Calinski-Harabasz': calinski_harabasz_score(X_scaled, labels_agg)
    }
    metrics_summary.append(metrics)

    if np.sum(core_mask) > 1 and len(np.unique(labels_db_core)) >= 2:
        metrics = {
            'Датасет': name,
            'Метод': 'DBSCAN',
            'Silhouette': silhouette_score(X_db_core, labels_db_core),
            'Davies-Bouldin': davies_bouldin_score(X_db_core, labels_db_core),
            'Calinski-Harabasz': calinski_harabasz_score(X_db_core, labels_db_core)
        }
    else:
        metrics = {
            'Датасет': name,
            'Метод': 'DBSCAN',
            'Silhouette': np.nan,
            'Davies-Bouldin': np.nan,
            'Calinski-Harabasz': np.nan
        }
    metrics_summary.append(metrics)

    # ------------------------------------------
    # 5. Визуализация
    # ------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # Исходные данные
    axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    axes[0].set_title('Исходные данные')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(alpha=0.3)

    # KMeans
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_km, s=10, cmap='tab10', alpha=0.8)
    axes[1].set_title(f'KMeans (k={best_k_kmeans})')
    axes[1].set_xlabel('x')
    axes[1].grid(alpha=0.3)

    # Agglomerative
    axes[2].scatter(X[:, 0], X[:, 1], c=labels_agg, s=10, cmap='tab10', alpha=0.8)
    axes[2].set_title(f'Agglomerative (k={best_k_agg})')
    axes[2].set_xlabel('x')
    axes[2].grid(alpha=0.3)

    # DBSCAN
    axes[3].scatter(X[:, 0], X[:, 1], c=labels_db, s=10, cmap='tab10', alpha=0.8)
    axes[3].set_title(f'DBSCAN (eps={best_eps:.2f}, min={best_ms})')
    axes[3].set_xlabel('x')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{name}_clusters.png', dpi=150)
    plt.show()

# ----------------------------------------------
# 6. Сводная таблица метрик
# ----------------------------------------------
df_metrics = pd.DataFrame(metrics_summary)
print("\n" + "="*80)
print("Сводка метрик:")
print(df_metrics.to_string(index=False))
