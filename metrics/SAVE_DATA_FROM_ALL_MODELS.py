import tensorflow as tf
import numpy as np
import cv2
import os
import json
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

from glob import glob

LABELS_PATH = "label_map.pbtxt"
BASE_MODEL_DIR = "/media/ivliev/DISK/job_reserch"
TEST_IMAGES_DIR = "/media/ivliev/DISK/job_reserch/main_research_data/training_demo/test_images"
/media/ivliev/DISK/job_reserch/main_research_data/training_demo/images/images_resized
CONFIDENCE_THRESHOLD = 0.5

# Загрузка разметки классов
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)

# Хранилище результатов
all_results = {}

Global_intersection = []
Global_union = []
Copy_flag = False

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

# Поиск всех research_* директорий
# research_dirs = sorted(glob(os.path.join(BASE_MODEL_DIR, "reserch__*/training_demo/exported_models/my_model/saved_model")))

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


    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]

    for count, image_name in enumerate(image_files):
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        xml_path = os.path.join(TEST_IMAGES_DIR, image_name.split('.')[0] + ".xml")

        # Загружаем изображение и разметку
        image_np, true_boxes, true_classes = load_image_and_xml(image_path, xml_path)
        if image_np is None:
            continue

        # Подготавливаем входные данные для модели
        input_data = tf.convert_to_tensor(image_np)
        input_data = input_data[tf.newaxis, ...]

        # Запускаем детекцию
        detections = model(input_data)



# Сохраняем JSON
with open("detection data.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\n✅ Готово! Метрики сохранены в metrics_main.json")