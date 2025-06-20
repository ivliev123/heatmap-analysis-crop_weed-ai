# Libraries
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import re

# Open an image from a computer 
def open_image_local(path_to_image):
    image = Image.open(path_to_image).convert("RGB") # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output

def calk_point_inside_box(data_point_array, box):
    x_min = box[0]
    y_min = box[1]
    x_max = box[2]
    y_max = box[3]

    crop_count = 0
    weed_count = 0 
    for point in data_point_array:
        x, y = point["coord"]
        class_ = point["class"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            if(class_ == "crop"):
                crop_count +=1
            if(class_ == "weed"):
                weed_count +=1

    return crop_count, weed_count


def calc_segments(x_size, y_size, a_size, data_point_array):
    
    x_seg = int(x_size / a_size)
    y_seg = int(y_size / a_size)

    seg_midl_array = []
    seg_data_array = []

    for j in range(y_seg):
        for i in range(x_seg): 
            x_seg_min = i * a_size
            y_seg_min = j * a_size
            x_seg_max = (i + 1) * a_size
            y_seg_max = (j + 1) * a_size
            x_seg_midl = x_seg_min + (x_seg_max - x_seg_min)/2
            y_seg_midl = y_seg_min + (y_seg_max - y_seg_min)/2
        
            seg_midl_array.append([x_seg_midl, y_seg_midl])

            box = [x_seg_min, y_seg_min, x_seg_max, y_seg_max]
            crop_count, weed_count = calk_point_inside_box(data_point_array, box)

            seg_data_array.append([x_seg_midl, y_seg_midl, crop_count, weed_count])

    return seg_data_array


# Извлекаем список имен файлов и сортируем по числовому индексу
def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1


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

detection_data_path = "detection_data.json"
# путь к папке с изображениями и файлам разметки
image_path_ = "main_research_data/training_demo/images/images_resized/"

path_to_save_figure = "3_FIGURE/"

a_size = 0.2

num_img_x = 20
num_img_y = 20
separetion_size = 0
img_size = [1137.0, 640.0] 
img_scale_x = 1 
img_scale_y = 640.0/1137.0 
max_size_x = num_img_x * img_scale_x + (num_img_x + 1) *  separetion_size
max_size_y = num_img_y * img_scale_y + (num_img_y + 1) *  separetion_size

print(max_size_x)



with open(detection_data_path, 'r') as file:
    data = json.load(file)





def make_seg_map_reserch(reserch__num):

    # основная диаграмма // Диаграмма на которой будут отображаться точки
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.set_xlim(0, max_size_x)
    ax.set_ylim(0, max_size_y)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("")

    data_point_array = []

    model_name = Model_Name_Dict[reserch__num]


    reserch_i = data[reserch__num]["detections"]
    model_data = data[reserch__num]["detections"]

    sorted_images = sorted(model_data.keys(), key=extract_number)

    index = 0  # например, первое изображение
    image_name = sorted_images[index]
    detections = model_data[image_name]

    print(len(sorted_images))

    for j in range(num_img_y):
        for i in range(num_img_x):


            image_index = j * num_img_x + i
            image_name = sorted_images[image_index]
            image_detections = model_data[image_name]


            image_path = image_path_ + image_name
            image = open_image_local(image_path)

            # Позиция изображения относительно глобальных координат
            x0 = separetion_size + (img_scale_x + separetion_size) * i
            x1 = separetion_size + img_scale_x + (img_scale_x + separetion_size) * i
            y0 = separetion_size + (img_scale_y + separetion_size) * j
            y1 = separetion_size + img_scale_y + (img_scale_y + separetion_size) * j

            # sys.exit(1)
            # Позиция детектируемого объета относительно изображения 
            for obj in range(len(image_detections)):
                box = image_detections[obj]['box']
                x_min = box['xmin']
                y_min = box['ymin']
                x_max = box['xmax']
                y_max = box['ymax']
                obj_box = [x_min, y_min, x_max, y_max]
                obj_class = image_detections[obj]['class']

                obj_x =                 (obj_box[0] + (obj_box[2] - obj_box[0])/2) / img_size[0] * img_scale_x
                obj_y =  (img_size[1] - (obj_box[1] + (obj_box[3] - obj_box[1])/2)) / img_size[1] * img_scale_y
                
                # print(obj_x, obj_y)

                # Позиция детектируемого объета относительно глобальных координат
                x_ = x0 + obj_x
                y_ = y0 + obj_y

                data_point_array.append({"class": obj_class, "coord": [x_, y_]})

                if(obj_class == "crop"):
                    plt.plot(x_, y_, marker='o', color='#00FF00')

                if(obj_class == "weed"):
                    plt.plot(x_, y_, marker='o', color='#FF0000')


            # Define the position for the image axes
            ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')


    seg_data_array = calc_segments(max_size_x, max_size_y, a_size, data_point_array)


    X = []
    Y = []
    Z_crop = []
    Z_weed = []

    for seg in seg_data_array:
        x, y, crop, weed = seg
        X.append(x)
        Y.append(y)
        Z_crop.append(crop)
        Z_weed.append(weed)


    # Display the plot
    plt.savefig(f"{path_to_save_figure}6_{reserch__num}_{model_name}_sim_map_.png", dpi=300)
    # plt.show()


    fig_crop, ax_crop = plt.subplots(figsize=(10, 5.5))

    sc_crop = ax_crop.scatter(X, Y, c=Z_crop, cmap='Greens', s=100, marker='s', vmin=0, vmax=4)
    ax_crop.set_title("Crop Heatmap")
    fig_crop.colorbar(sc_crop, ax=ax_crop)

    ax_crop.set_xlim(0, max_size_x)
    ax_crop.set_ylim(0, max_size_y)
    ax_crop.set_xlabel("X")
    ax_crop.set_ylabel("Y")
    ax_crop.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{path_to_save_figure}6_{reserch__num}_{model_name}_sim_map_crop_heatmap.png", dpi=300)
    plt.close(fig_crop)


    fig_weed, ax_weed = plt.subplots(figsize=(10, 5.5))

    sc_weed = ax_weed.scatter(X, Y, c=Z_weed, cmap='Reds', s=100, marker='s', vmin=0, vmax=10)
    ax_weed.set_title("Weed Heatmap")
    fig_weed.colorbar(sc_weed, ax=ax_weed)

    ax_weed.set_xlim(0, max_size_x)
    ax_weed.set_ylim(0, max_size_y)
    ax_weed.set_xlabel("X")
    ax_weed.set_ylabel("Y")
    ax_weed.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{path_to_save_figure}6_{reserch__num}_{model_name}_sim_map_weed_heatmap.png", dpi=300)
    plt.close(fig_weed)


for i in range(1,12):
    reserch__num = f"reserch__{i}"
    make_seg_map_reserch(reserch__num)