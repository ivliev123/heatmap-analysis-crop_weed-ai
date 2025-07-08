from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# --- Загрузка изображения и аннотаций ---
def open_image(path):
    return np.array(Image.open(path).convert("RGB"))

def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes, labels = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(label)
    return boxes, labels

# --- Подсчёт количества объектов внутри сегмента ---
def count_objects_in_box(points, box):
    x_min, y_min, x_max, y_max = box
    crop, weed = 0, 0
    for p in points:
        x, y = p["coord"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            crop += p["class"] == "crop"
            weed += p["class"] == "weed"
    return crop, weed

# --- Генерация сегментов и подсчёт ---
def compute_segments(img_w, img_h, grid_size, points):
    num_x = int(img_w / grid_size)
    num_y = int(img_h / grid_size)
    segments = []

    for j in range(num_y):
        for i in range(num_x):
            x0 = i * grid_size
            y0 = j * grid_size
            x1 = x0 + grid_size
            y1 = y0 + grid_size
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            crop, weed = count_objects_in_box(points, [x0, y0, x1, y1])
            segments.append((cx, cy, crop, weed))
    return segments

# --- Основной код ---

# Параметры
# path = "../../../crop_weed_research_data/images/test/"
image_dir = "../../../crop_weed_research_data/images/train/"
path_to_save_figure = "2_FIGURE/"
# data = [14, 392, 847, 1052, 2957, 4147, 4195, 4219, 4170]
image_ids = [3795, 3798, 3808, 3812, 3816, 3824, 3869, 3882, 3889]

img_size = (1137.0, 640.0)
grid_step = 0.2
img_scale_x = 1.0
img_scale_y = img_size[1] / img_size[0]
num_cols, num_rows = 3, 3
separation = 0.01
canvas_w = num_cols * img_scale_x + (num_cols + 1) * separation
canvas_h = num_rows * img_scale_y + (num_rows + 1) * separation

# Сбор точек объектов
points = []

# fig, ax = plt.subplots(figsize=(10, 5.5))
# ax.set_xlim(0, canvas_w)
# ax.set_ylim(0, canvas_h)
# ax.set_title("Этап 4: Объекты на изображениях")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")

for j in range(num_rows):
    for i in range(num_cols):
        idx = j * num_cols + i
        img_id = image_ids[idx]
        img_path = f"{image_dir}image_{img_id}.jpg"
        xml_path = f"{image_dir}image_{img_id}.xml"

        image = open_image(img_path)
        boxes, labels = load_annotations(xml_path)

        # Положение на общей диаграмме
        x0 = separation + (img_scale_x + separation) * i
        y0 = separation + (img_scale_y + separation) * j
        x1 = x0 + img_scale_x
        y1 = y0 + img_scale_y

        # ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')

        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            center_x = (xmin + xmax) / 2 / img_size[0] * img_scale_x + x0
            center_y = (img_size[1] - (ymin + ymax) / 2) / img_size[1] * img_scale_y + y0
            points.append({"class": label, "coord": (center_x, center_y)})

            color = "#00FF00" if label == "crop" else "#FF0000"
            # ax.plot(center_x, center_y, marker='o', color=color)

# plt.tight_layout()
# plt.savefig(f"{save_dir}4_etap.png", dpi=300)
# plt.close(fig)

# --- Построение тепловых карт ---
segments = compute_segments(canvas_w, canvas_h, grid_step, points)
X, Y, Z_crop, Z_weed = zip(*segments)

# Карта для crop
fig_crop, ax_crop = plt.subplots(figsize=(10, 5.5))
sc = ax_crop.scatter(X, Y, c=Z_crop, cmap='Greens', s=300, marker='s')
ax_crop.set_title("Crop Heatmap")
ax_crop.set_xlim(0, canvas_w)
ax_crop.set_ylim(0, canvas_h)
ax_crop.set_xlabel("X, m")
ax_crop.set_ylabel("Y, m")
ax_crop.set_aspect('equal')
fig_crop.colorbar(sc, ax=ax_crop)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}4_etap_crop_heatmap.png", dpi=300)
plt.close(fig_crop)

# Карта для weed
fig_weed, ax_weed = plt.subplots(figsize=(10, 5.5))
sc = ax_weed.scatter(X, Y, c=Z_weed, cmap='Reds', s=300, marker='s')
ax_weed.set_title("Weed Heatmap")
ax_weed.set_xlim(0, canvas_w)
ax_weed.set_ylim(0, canvas_h)
ax_weed.set_xlabel("X, m")
ax_weed.set_ylabel("Y, m")
ax_weed.set_aspect('equal')
fig_weed.colorbar(sc, ax=ax_weed)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}4_etap_weed_heatmap.png", dpi=300)
plt.close(fig_weed)
