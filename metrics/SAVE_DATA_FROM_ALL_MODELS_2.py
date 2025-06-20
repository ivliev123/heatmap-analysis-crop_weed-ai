import tensorflow as tf
import numpy as np
import cv2
import os
import json
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from glob import glob
import gc

LABELS_PATH = "label_map.pbtxt"
BASE_MODEL_DIR = "/media/ivliev/DISK/job_reserch"
# TEST_IMAGES_DIR = "/media/ivliev/DISK/job_reserch/main_research_data/training_demo/test_images"
TEST_IMAGES_DIR = "/media/ivliev/DISK/job_reserch/main_research_data/training_demo/images/images_resized"
CONFIDENCE_THRESHOLD = 0.5

# Загрузка разметки классов
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)

# Хранилище результатов
all_results = {}

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

# Получаем список моделей
research_dirs = [f"{BASE_MODEL_DIR}/reserch__{i+1}/training_demo/exported_models/my_model/saved_model" for i in range(11)]

def load_image_and_xml(image_path, xml_path):
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загрузка и разбор XML разметки
        boxes = []
        classes = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append([ymin, xmin, ymax, xmax])
                classes.append(class_name)
        return image_rgb, boxes, classes
    except:
        return None, [], []

for model_path in research_dirs:
    tf.keras.backend.clear_session()
    gc.collect()
    
    model_numb = model_path.split("/")[5]  # "reserch__X"
    print(f"\n🔍 Обработка модели: {model_numb} — {Model_Name_Dict[model_numb]}")

    # Загрузка модели
    detect_fn = tf.saved_model.load(model_path)
    model_results = {}

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for image_name in image_files:
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        xml_path = os.path.join(TEST_IMAGES_DIR, os.path.splitext(image_name)[0] + ".xml")

        image_np, true_boxes, true_classes = load_image_and_xml(image_path, xml_path)
        if image_np is None:
            continue

        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

        # Получение предсказаний
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_scores = detections['detection_scores'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)

        height, width, _ = image_np.shape

        filtered_results = []
        for i in range(num_detections):
            if detection_scores[i] >= CONFIDENCE_THRESHOLD:
                box = detection_boxes[i]
                y_min, x_min, y_max, x_max = box
                box_pixels = {
                    "xmin": int(x_min * width),
                    "ymin": int(y_min * height),
                    "xmax": int(x_max * width),
                    "ymax": int(y_max * height),
                }
                class_id = detection_classes[i]
                class_name = category_index[class_id]['name'] if class_id in category_index else "unknown"
                filtered_results.append({
                    "box": box_pixels,
                    "score": float(detection_scores[i]),
                    "class": class_name
                })

        model_results[image_name] = filtered_results

    all_results[model_numb] = {
        "model_name": Model_Name_Dict[model_numb],
        "detections": model_results
    }

# Сохраняем результат в JSON
with open("detection_data_2.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print("✅ Детектирование завершено. Результаты сохранены в 'detection_data.json'.")
