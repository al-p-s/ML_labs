import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image

image_path = "mgk.jpg"
img = Image.open(image_path)
img = img.convert("RGB")
img_array = np.array(img)

h, w, d = img_array.shape
pixels = img_array.reshape(h * w, d)

sample_size = min(20000, h * w)
pixels_sample = shuffle(pixels, random_state=42)[:sample_size]

n_colors = 8
kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
kmeans.fit(pixels_sample)
centers = kmeans.cluster_centers_.astype(np.uint8)

labels = kmeans.predict(pixels)
compressed_pixels = centers[labels]
compressed_img = compressed_pixels.reshape(h, w, d)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_array)
axes[0].set_title("original")
axes[0].axis("off")

axes[1].imshow(compressed_img)
axes[1].set_title(f"compressed palette ({n_colors} colors)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("color_compression.png", dpi=150)
plt.show()
