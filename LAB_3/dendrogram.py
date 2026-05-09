import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df = pd.read_csv('z_datasets/votes.csv', index_col=0)  # первая колонка — названия штатов
df.index = [str(i) for i in range(1, len(df) + 1)]
print(f"Размер данных: {df.shape}")
print("Пропуски (NA):", df.isna().sum().sum())

# 2. Замена строковых 'NA' на NaN и преобразование к числовому типу
df.replace('NA', np.nan, inplace=True)
df = df.astype(float)

# 3. Заполнение пропусков средним по году (столбцу)
#    Это позволяет сохранить все 50 штатов
df_filled = df.apply(lambda col: col.fillna(col.mean()), axis=0)

# Проверка отсутствия пропусков
assert df_filled.isna().sum().sum() == 0

# 4. Стандартизация (важно для евклидова расстояния)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filled)

# 5. Иерархическая кластеризация (метод Варда)
Z = linkage(X_scaled, method='ward')

# 6. Визуализация дендрограммы
plt.figure(figsize=(14, 7))
dendrogram(Z, labels=df_filled.index, leaf_rotation=90, leaf_font_size=8,
           color_threshold=12)  # порог для окрашивания кластеров
plt.title('Дендрограмма штатов по голосованию за республиканцев')
plt.xlabel('Штаты')
plt.ylabel('Евклидово расстояние')
plt.tight_layout()
plt.savefig('votes_dendrogram.png', dpi=150)
plt.show()
