import json
import matplotlib.pyplot as plt

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

path_to_save_figure = "1_FIGURE/"


 

# Цвета и метки
colors = [
    "red", "limegreen", "darkgreen", "blue", "magenta",
    "navy", "deeppink", "orange", "purple", "gray", "brown"
]
markers = ["o", "o", "o", "o", "o", "s", "s", "s", "s", "D", "D"]

# Построение графика



# Выбор метрики
y_metric = "precision"  
x_metric = "TIME"   
plt.figure(figsize=(8, 6))

for i in range(len(Model_Name_Dict)):
    reserch_num = f"reserch__{i+1}"
    model_name = Model_Name_Dict[reserch_num]
    print(model_name)
    x = other_data[model_name][x_metric] * 1000  # Переводим секунды в миллисекунды
    y = main_data[model_name][y_metric] * 100
    plt.scatter(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=150, label=model_name)

plt.grid(True, linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.xlabel("Time, ms.")
plt.ylabel("Precision, %")
plt.legend(fontsize=10, loc="lower right")
# plt.ylim(40, 75)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}1_Precision.png", dpi=300)
plt.close()


# Выбор метрики
y_metric = "recall"  
x_metric = "TIME"   
plt.figure(figsize=(8, 6))

for i in range(len(Model_Name_Dict)):
    reserch_num = f"reserch__{i+1}"
    model_name = Model_Name_Dict[reserch_num]
    print(model_name)
    x = other_data[model_name][x_metric] * 1000  # Переводим секунды в миллисекунды
    y = main_data[model_name][y_metric] * 100
    plt.scatter(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=150, label=model_name)

plt.grid(True, linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.xlabel("Time, ms.")
plt.ylabel("Recall, %")
plt.legend(fontsize=10, loc="lower right")
# plt.ylim(40, 75)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}2_Recall.png", dpi=300)
plt.close()



y_metric = "MEMORY"  
x_metric = "TIME" 
plt.figure(figsize=(8, 6))

for i in range(len(Model_Name_Dict)):
    reserch_num = f"reserch__{i+1}"
    model_name = Model_Name_Dict[reserch_num]
    print(model_name)
    x = other_data[model_name][x_metric] * 1000  # Переводим секунды в миллисекунды
    y = other_data[model_name][y_metric]
    plt.scatter(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=150, label=model_name)

plt.grid(True, linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.xlabel("Time, ms.")
plt.ylabel("Memory, Mb")
plt.legend(fontsize=10, loc="lower right")
# plt.ylim(40, 75)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}3_Memory.png", dpi=300)
plt.close()


y_metric = "GFLOPS"  
x_metric = "TIME" 
plt.figure(figsize=(8, 6))

for i in range(len(Model_Name_Dict)):
    reserch_num = f"reserch__{i+1}"
    model_name = Model_Name_Dict[reserch_num]
    print(model_name)
    x = other_data[model_name][x_metric] * 1000  # Переводим секунды в миллисекунды
    y = other_data[model_name][y_metric]
    plt.scatter(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=150, label=model_name)

plt.grid(True, linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.xlabel("Time, ms.")
plt.ylabel("FLOPS 10^9")
plt.legend(fontsize=10, loc="lower right")
# plt.ylim(40, 75)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}4_FLOPS.png", dpi=300)
plt.close()



y_metric = "PARAMS"  
x_metric = "TIME" 
plt.figure(figsize=(8, 6))

for i in range(len(Model_Name_Dict)):
    reserch_num = f"reserch__{i+1}"
    model_name = Model_Name_Dict[reserch_num]
    print(model_name)
    x = other_data[model_name][x_metric] * 1000  # Переводим секунды в миллисекунды
    y = other_data[model_name][y_metric] / 1000000
    plt.scatter(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=150, label=model_name)

plt.grid(True, linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.xlabel("Time, ms.")
plt.ylabel("Params, 10^6")
plt.legend(fontsize=10, loc="lower right")
# plt.ylim(40, 75)
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}5_PARAMS.png", dpi=300)
plt.close()




