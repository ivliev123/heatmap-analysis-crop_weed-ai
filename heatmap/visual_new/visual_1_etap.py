from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def open_image_local(path_to_image):
    image = Image.open(path_to_image).convert("RGB")
    return np.array(image)

# Paths
# path = "../../../crop_weed_research_data/images/test/"
path = "../../../crop_weed_research_data/images/train/"
path_to_save_figure = "2_FIGURE/"
# data = [14, 392, 847, 1052, 2957, 4147, 4195, 4219, 4170]
data = [3795, 3798, 3808, 3812, 3816, 3824, 3869, 3882, 3889]

# Plot settings
fig, ax = plt.subplots(figsize=(12, 8))

num_img_x, num_img_y = 3, 3
separation = 0.01
img_scale_x = 1
img_scale_y = 640.0 / 1137.0
max_size_x = num_img_x * img_scale_x + (num_img_x + 1) * separation
max_size_y = num_img_y * img_scale_y + (num_img_y + 1) * separation

ax.set_xlim(0, max_size_x)
ax.set_ylim(0, max_size_y)
ax.set_xlabel("X, m")
ax.set_ylabel("Y, m")
# ax.set_title("Stage 1: Data collection")

# Place images
for j in range(num_img_y):
    for i in range(num_img_x):
        idx = j * num_img_x + i
        image_path = f"{path}image_{data[idx]}.jpg"
        image = open_image_local(image_path)

        x0 = separation + (img_scale_x + separation) * i
        x1 = x0 + img_scale_x
        y0 = separation + (img_scale_y + separation) * j
        y1 = y0 + img_scale_y

        ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')

plt.tight_layout()
plt.savefig(f"{path_to_save_figure}1_etap.png", dpi=300)
# plt.show()
