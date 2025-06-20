import numpy as np
import matplotlib.pyplot as plt
import json

# Загрузка данных
with open("metrics_main.json", "r") as f:
    main_data = json.load(f)

with open("metrics_other.json", "r") as f:
    other_data = json.load(f)

Model_Name_Dict = {
    "reserch__1": "SSD MobileNet V1 FPN 640x640",
    "reserch__2": "SSD MobileNet V2 FPNLite 640x640",
    "reserch__3": "SSD ResNet50 V1 FPN 640x640",
    "reserch__4": "SSD ResNet101 V1 FPN 640x640",
    "reserch__5": "SSD ResNet152 V1 FPN 640x640",
    "reserch__6": "Faster R-CNN ResNet50 V1 640x640",
    "reserch__7": "Faster R-CNN ResNet101 V1 640x640",
    "reserch__8": "Faster R-CNN ResNet152 V1 640x640",
    "reserch__9": "Faster R-CNN Inception ResNet V2 640x640",
    "reserch__10": "EfficientDet D0 512x512",
    "reserch__11": "EfficientDet D1 640x640"
}

# Метрики, которые используем на лепестковой диаграмме
metrics = [
    ("iou", "IoU"),
    ("recall", "Recall"),
    ("precision", "Precision"),
    ("iou_2.0", "IoU@2.0"),
    ("TIME", "Time (inv)"),
    ("MEMORY", "Memory (inv)"),
    ("GFLOPS", "GFLOPS"),
    ("PARAMS", "Params (inv)")
]

# Список моделей, которые хотим отобразить
selected_models = [
    "SSD MobileNet V2 FPNLite 640x640",
    "Faster R-CNN ResNet101 V1 640x640",
    "EfficientDet D0 512x512"
]

# Нормализация метрик для корректного отображения
def normalize(values, inverse=False):
    values = np.array(values)
    if inverse:
        values = np.max(values) - values + np.min(values)
    return (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)

# Подготовка данных
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # для замыкания графика

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for model in selected_models:
    values = []
    for key, _ in metrics:
        if key in main_data[model]:
            val = main_data[model][key]
            inv = False
        else:
            val = other_data[model][key]
            inv = key in ["TIME", "MEMORY", "PARAMS"]  # Инверсные метрики
        values.append((val, inv))

    # Нормализация значений
    normalized = [normalize([v[0] for v in values], inverse=v[1])[0] for v in values]
    normalized += normalized[:1]  # замыкаем круг

    ax.plot(angles, normalized, label=model)
    ax.fill(angles, normalized, alpha=0.1)

# Настройка осей
metric_labels = [label for _, label in metrics]
metric_labels += metric_labels[:1]
ax.set_xticks(angles)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_yticklabels([])
ax.set_title("Model Radar Chart", size=14, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("1_FIGURE/radar_chart.png", dpi=300)
plt.show()
