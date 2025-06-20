import tensorflow as tf
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import time
import psutil
import json


from object_detection.utils import label_map_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Пути к папкам
LABELS_PATH = "label_map.pbtxt"
BASE_MODEL_DIR = "/media/ivliev/DISK/job_reserch"
TEST_IMAGES_DIR = "/media/ivliev/DISK/job_reserch/main_research_data/training_demo/test_images"
CONFIDENCE_THRESHOLD = 0.5
MAX_IMAGES = 431

# Загружаем метки
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)
print(category_index)


# Хранилище результатов
all_results = {}


# Загрузка изображения и аннотации
def load_image_and_xml(image_path, xml_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return None, None, None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    true_boxes = []
    true_classes = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        true_boxes.append((xmin, ymin, xmax, ymax))
        true_classes.append(class_name)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), true_boxes, true_classes

# Расчёт FPS
def calc_fps():
    images = []

    for image_name in image_files[:MAX_IMAGES]:
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        xml_path = os.path.join(TEST_IMAGES_DIR, os.path.splitext(image_name)[0] + ".xml")

        image_np, _, _ = load_image_and_xml(image_path, xml_path)
        if image_np is None:
            continue

        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
        images.append(input_tensor)

    start_time = time.time()
    for image_tensor in images:
        _ = model(image_tensor)
    end_time = time.time()

    fps = len(images) / (end_time - start_time)
    print(f"FPS: {fps:.2f}")
    return fps

# Расчёт использования памяти (RAM)
def calc_memory():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 2)  # В MB
    print(f"Memory usage (RAM): {memory_usage:.2f} MB")
    return memory_usage

# Расчёт FLOPs
def calc_flops(model):
    concrete_func = model.signatures["serving_default"]
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )

        if flops is not None:
            print(f"FLOPs: {flops.total_float_ops / 1e9:.2f} GFLOPs")
        else:
            print("Не удалось рассчитать FLOPs.")
            
    return flops.total_float_ops / 1e9

# Расчёт количества параметров
def calc_params(model):
    concrete_func = model.signatures["serving_default"]
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    total_params = 0

    for node in frozen_func.graph.as_graph_def().node:
        if 'Const' in node.op and node.attr['value'].tensor.tensor_content:
            shape = node.attr['value'].tensor.tensor_shape
            dims = [dim.size for dim in shape.dim]
            param_count = np.prod(dims) if dims else 0
            total_params += param_count

    print(f"Total parameters (approx): {total_params / 1e6:.2f}M")
    return total_params



Model_Name_Dict = { "reserch__1" : "SSD MobileNet V1 FPN 640x640",
                    "reserch__2" : "SSD MobileNet V2 FPNLite 640x640",
                    "reserch__3" : "SSD ResNet50 V1 FPN 640x640",
                    "reserch__4" : "SSD ResNet101 V1 FPN 640x640",
                    "reserch__5" : "SSD ResNet152 V1 FPN 640x640",
                    "reserch__6" : "Faster R-CNN ResNet50 V1 640x640",
                    "reserch__7" : "Faster R-CNN ResNet101 V1 640x640",
                    "reserch__8" : "Faster R-CNN ResNet152 V1 640x640",
                    "reserch__9" : "Faster R-CNN Inception ResNet V2 640x640",
                    "reserch__10" : "EfficientDet D0 512x512",
                    "reserch__11" : "EfficientDet D1 640x640"}


research_dirs = []
for i in range(11):
    model_path = BASE_MODEL_DIR + f"/reserch__{i+1}/training_demo/exported_models/my_model/saved_model"
    research_dirs.append(model_path)

# research_dirs = research_dirs[0:3] # временная заглушка для отладки 
print(research_dirs)



for model_path in research_dirs:
    model_numb = model_path.split("/")[5]  # например "reserch__10"
    print(f"\n🔍 Обработка модели: {model_numb}")

    model = tf.saved_model.load(model_path)

    # Получение списка изображений
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # # Запуск метрик
    fps = calc_fps()
    memory = calc_memory()
    flops = calc_flops(model)  # Раскомментируй, если tf.profiler поддерживает это для твоей модели
    params = calc_params(model)
    time_ = 1/fps

    print(f"FPS: {fps}; MEMORY: {memory}; GFLOPS: {flops}; PARAMS: {params}; TIME: {time_}")
    # print("Complete")

    model_name = Model_Name_Dict[model_numb]

    # all_results[model_name] = {
    #                             "FPS": round(fps, 4),
    #                             "MEMORY": round(memory, 4),
    #                             "GFLOPS": round(flops, 4),
    #                             "PARAMS": round(params, 4),
    #                             "TIME": round(time_, 4)
    #                             }
# # Сохраняем JSON
# with open("metrics_other.json", "w") as f:
#     json.dump(all_results, f, indent=4)


    all_results[model_name] = {
                                "FPS": round(float(fps), 4),
                                "MEMORY": round(float(memory), 4),
                                "GFLOPS": round(float(flops), 4),
                                "PARAMS": round(float(params), 4),
                                "TIME": round(float(time_), 4)
                            }
def convert_np(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError

with open("metrics_other.json", "w") as f:
    json.dump(all_results, f, indent=4, default=convert_np)


print("\n✅ Готово! Метрики сохранены в metrics_other.json")