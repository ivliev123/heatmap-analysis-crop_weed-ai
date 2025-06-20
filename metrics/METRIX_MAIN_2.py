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
CONFIDENCE_THRESHOLD = 0.5

# Загрузка разметки классов
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)

# Хранилище результатов
all_results = {}

Global_intersection = []
Global_union = []
Copy_flag = False


def load_image_and_xml(image_path, xml_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return None, None, None  # Теперь возвращаем три значения

    tree = ET.parse(xml_path)
    root = tree.getroot()

    true_boxes = []
    true_classes = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text  # Извлекаем название класса объекта
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        true_boxes.append((xmin, ymin, xmax, ymax))
        true_classes.append(class_name)  # Добавляем класс в список

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), true_boxes, true_classes


# def draw_ground_truth_and_pred(image_np, true_boxes, pred_boxes, scores, threshold=CONFIDENCE_THRESHOLD):
def draw_ground_truth_and_pred(image_np, detections, true_boxes, threshold=CONFIDENCE_THRESHOLD):
    
    pred_boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    """
    Визуализирует рамки из разметки (зеленые), предсказанные боксы (красные) и их пересечение (фиолетовый).
    """
    overlay = image_np.copy()
    h, w, _ = image_np.shape

    # Масштабируем предсказанные боксы из нормализованных координат
    pred_boxes = np.array([
        [int(box[1] * w), int(box[0] * h), int(box[3] * w), int(box[2] * h)] for box in pred_boxes
    ])

    # Отрисовка разметки (зеленый)
    for box in true_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Отрисовка предсказанных боксов (желтый)
    for i, box in enumerate(pred_boxes):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Проверяем пересечение с разметкой
            for true_box in true_boxes:
                ix1 = max(true_box[0], x1)
                iy1 = max(true_box[1], y1)
                ix2 = min(true_box[2], x2)
                iy2 = min(true_box[3], y2)

                # Если есть пересечение, закрашиваем фиолетовым
                if ix1 < ix2 and iy1 < iy2:
                    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (255, 0, 255), -1)

    # Наложение фиолетового с прозрачностью 0.5
    cv2.addWeighted(overlay, 0.4, image_np, 0.6, 0, image_np)

    return image_np


def draw_boxes(image_np, detections, threshold=CONFIDENCE_THRESHOLD):
    pred_boxes_vis = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
        
    vis_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        # pred_boxes,
        pred_boxes_vis,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=threshold)
    return image_np



def calculate_iou(image_np, detections, true_boxes, true_classes, threshold=CONFIDENCE_THRESHOLD):
    h, w, _ = image_np.shape

    pred_boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    pred_classes = detections['detection_classes'][0].numpy().astype(int)

    # Фильтруем боксы по threshold
    filtered_preds = [
        ([box[1] * w, box[0] * h, box[3] * w, box[2] * h], pred_classes[i])
        for i, box in enumerate(pred_boxes) if scores[i] >= threshold
    ]

    pred_boxes, pred_classes = zip(*filtered_preds) if filtered_preds else ([], [])
    pred_class_names = [
        category_index.get(cls, {'name': 'unknown'})['name'] for cls in pred_classes
    ]
    pred_classes = pred_class_names


    iou_scores = []
    for i, true_box in enumerate(true_boxes):
        max_iou = 0
        max_intersection = 0
        max_union = 0
        for j, pred_box in enumerate(pred_boxes):
            #Copy_flag = True
            iou, intersection, union = calculate_intersection(true_box, pred_box, True)
            #Copy_flag = False
            if iou > max_iou and pred_classes[j] == true_classes[i]:  # Учитываем правильность класса
                max_iou = iou
                max_intersection = intersection
                max_union = union
        iou_scores.append(max_iou)
        Global_intersection.append(max_intersection)
        Global_union.append(max_union)
    
    return iou_scores


def calculate_recall(image_np, detections, true_boxes, true_classes, iou_threshold=0.5):
    tp = 0
    fn = len(true_boxes)

    h, w, _ = image_np.shape
    pred_boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    pred_classes = detections['detection_classes'][0].numpy().astype(int)

    # Фильтрация по threshold
    filtered_preds = [
        ([box[1] * w, box[0] * h, box[3] * w, box[2] * h], pred_classes[i])
        for i, box in enumerate(pred_boxes) if scores[i] >= iou_threshold
    ]
    
    pred_boxes, pred_classes = zip(*filtered_preds) if filtered_preds else ([], [])
    pred_class_names = [
        category_index.get(cls, {'name': 'unknown'})['name'] for cls in pred_classes
    ]
    pred_classes = pred_class_names

    for i, true_box in enumerate(true_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou, _, _ = calculate_intersection(true_box, pred_box, False)
            if iou > iou_threshold and pred_classes[j] == true_classes[i]:  # Проверяем класс
                tp += 1
                fn -= 1
                break

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall


def calculate_precision(image_np, detections, true_boxes, true_classes, iou_threshold=0.5):
    tp = 0
    fp = 0

    h, w, _ = image_np.shape
    pred_boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    pred_classes = detections['detection_classes'][0].numpy().astype(int)

    # Фильтрация по threshold
    filtered_preds = [
        ([box[1] * w, box[0] * h, box[3] * w, box[2] * h], pred_classes[i])
        for i, box in enumerate(pred_boxes) if scores[i] >= iou_threshold
    ]

    pred_boxes, pred_classes = zip(*filtered_preds) if filtered_preds else ([], [])
    pred_class_names = [
        category_index.get(cls, {'name': 'unknown'})['name'] for cls in pred_classes
    ]
    pred_classes = pred_class_names

    for j, pred_box in enumerate(pred_boxes):
        matched = False
        for i, true_box in enumerate(true_boxes):
            iou, _, _ = calculate_intersection(true_box, pred_box, False)
            if iou > iou_threshold and pred_classes[j] == true_classes[i]:  #Проверяем класс
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1  # Предсказали, но нет совпадения — это FP

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision


def calculate_intersection(box1, box2, flag):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    else:
        intersection_area = 0

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
        
    return iou, intersection_area, union_area



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
    # iou_scores_all = []
    # total_inter = 0
    # total_union = 0
    # recalls = []
    # precisions = []

    iou_array = []
    recall_array = []
    precision_array = []

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

        # Рассчитываем метрики
        iou_scores = calculate_iou(image_np, detections, true_boxes, true_classes)
        iou_scores_mean = np.mean(iou_scores)
        recall = calculate_recall(image_np, detections, true_boxes, true_classes)
        precision = calculate_precision(image_np, detections, true_boxes, true_classes)

        iou_array.append(iou_scores_mean)
        recall_array.append(recall)
        precision_array.append(precision)

        # Рисуем боксы на изображении
        # image_with_boxes = draw_ground_truth_and_pred(image_np, detections, true_boxes)
        # image_with_boxes = draw_boxes(image_with_boxes, detections, true_boxes)


        avg_iou = np.mean(iou_array)
        avg_recall = np.mean(recall_array)
        avg_precision = np.mean(precision_array)
        avg_iou_2 = sum(Global_intersection)/sum(Global_union)


        model_name = Model_Name_Dict[model_numb]

        all_results[model_name] = {
                                    "iou": round(avg_iou, 4),
                                    "recall": round(avg_recall, 4),
                                    "precision": round(avg_precision, 4),
                                    "iou_2.0": round(avg_iou_2, 4)
                                    }
# Сохраняем JSON
with open("metrics_main.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\n✅ Готово! Метрики сохранены в metrics_main.json")