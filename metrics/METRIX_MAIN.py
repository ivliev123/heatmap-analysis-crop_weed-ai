import tensorflow as tf
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

Global_intersection = []
Global_union = []
Copy_flag = False

#path to folders
MODEL_DIR = "/media/ivliev/DISK/job_reserch/reserch_10/training_demo/exported_models/my_model/saved_model"
LABELS_PATH = "label_map.pbtxt"

INPUT_FOLDER = "/media/ivliev/DISK/job_reserch/main_research_data/training_demo/test_images"
OUTPUT_FOLDER = "test_output"
CONFIDENCE_THRESHOLD = 0.5

#creating output folder, if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#downloading model
print("Downloading model...")
model = tf.saved_model.load(MODEL_DIR)
print("Model downloaded!")

#downloading labels
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)
print(category_index)

#fun for dowloading images
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
    #if flag:
    #    Global_intersection.append(intersection_area)
    #    Global_union.append(union_area)
        
    return iou, intersection_area, union_area




image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith((".jpg", ".png", ".jpeg"))]

# Ограничиваем количество обрабатываемых изображений
#MAX_IMAGES = 10
iou_array = []
recall_array = []
precision_array = []

for count, image_name in enumerate(image_files):#[:MAX_IMAGES], start=1):
    image_path = os.path.join(INPUT_FOLDER, image_name)
    xml_path = os.path.join(INPUT_FOLDER, image_name.split('.')[0] + ".xml")

    # Загружаем изображение и разметку
    # image_np, true_boxes = load_image_and_xml(image_path, xml_path)
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
    image_with_boxes = draw_ground_truth_and_pred(image_np, detections, true_boxes)
    # image_with_boxes = draw_boxes(image_with_boxes, detections, true_boxes)


    # Выводим результаты
    print(f"For img: {image_name} \t IoU scores: {iou_scores} \t Recall: {recall} \t Precision: {precision}")

    # Сохраняем обработанное изображение

    # output_path = os.path.join(OUTPUT_FOLDER, image_name)
    # cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

    print(f"Processed {count} images...")

average_iou = np.mean(iou_array)
average_recall = np.mean(recall_array)
average_precision = np.mean(precision_array)

average_iou_2 = sum(Global_intersection)/sum(Global_union)

print(f"среднее иоу {average_iou} среднее рекол {average_recall} среднее пресижн {average_precision}")
print(f"AVAREGE 2.0 {average_iou_2}")
print("Complete")
