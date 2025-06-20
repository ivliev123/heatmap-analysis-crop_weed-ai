# Libraries
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Open an image from a computer 
def open_image_local(path_to_image):
    image = Image.open(path_to_image).convert("RGB") # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output


def load_xml_data(xml_path):
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

    return true_boxes, true_classes
    
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


def calc_segments(x_size, y_size, a_size):
    
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




# путь к папке с изображениями и файлам разметки
path = "main_research_data/training_demo/images/test/"
path_to_save_figure = "2_FIGURE/"

data = [    14, 392, 847,
            1052, 2957, 4147,
            4195, 4219, 4170,
    # Добавь остальные изображения по аналогии
]

a_size = 0.2

num_img_x = 3
num_img_y = 3
separetion_size = 0
img_size = [1137.0, 640.0] 
img_scale_x = 1 
img_scale_y = 640.0/1137.0 
max_size_x = num_img_x * img_scale_x + (num_img_x + 1) *  separetion_size
max_size_y = num_img_y * img_scale_y + (num_img_y + 1) *  separetion_size
print(max_size_x)


# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 5.5))
# основная диаграмма
ax.set_xlim(0, max_size_x)
ax.set_ylim(0, max_size_y)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Этап 4. ")

data_point_array = []

for j in range(num_img_y):
    for i in range(num_img_x):
        # Open the image from my computer

        # image = open_image_local(path + data[j * num_img_x + i]["path"])

        # image_path = path + "image_" +data[j * num_img_x + i] + ".jpg"

        image_path = f"{path}image_{data[j * num_img_x + i]}.jpg"
        # print(image_path)
        xml_path = f"{path}image_{data[j * num_img_x + i]}.xml"
        print(xml_path)

        image = open_image_local(image_path)

        true_boxes, true_classes = load_xml_data(xml_path)
        # print(true_boxes, true_classes)

        # Define the position and size parameters
        x0 = separetion_size + (img_scale_x + separetion_size) * i
        x1 = separetion_size + img_scale_x + (img_scale_x + separetion_size) * i
        y0 = separetion_size + (img_scale_y + separetion_size) * j
        y1 = separetion_size + img_scale_y + (img_scale_y + separetion_size) * j

        for obj in range(len(true_classes)):
            obj_class = true_classes[obj]
            obj_box = true_boxes[obj]
            obj_x =                 (obj_box[0] + (obj_box[2] - obj_box[0])/2) / img_size[0] * img_scale_x
            obj_y =  (img_size[1] - (obj_box[1] + (obj_box[3] - obj_box[1])/2)) / img_size[1] * img_scale_y
            
            # print(obj_x, obj_y)

            x_ = x0 + obj_x
            y_ = y0 + obj_y

            data_point_array.append({"class": obj_class, "coord": [x_, y_]})

            if(obj_class == "crop"):
                plt.plot(x_, y_, marker='o', color='#00FF00')
            if(obj_class == "weed"):
                plt.plot(x_, y_, marker='o', color='#FF0000')

        # Define the position for the image axes
        ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')

# Display the plot
plt.savefig("4_etap.png", dpi=300)
# plt.show()





seg_data_array = calc_segments(max_size_x, max_size_y, a_size)

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


fig_crop, ax_crop = plt.subplots(figsize=(10, 5.5))
sc_crop = ax_crop.scatter(X, Y, c=Z_crop, cmap='Greens', s=300, marker='s')
ax_crop.set_title("Crop Heatmap")
fig_crop.colorbar(sc_crop, ax=ax_crop)

ax_crop.set_xlim(0, max_size_x)
ax_crop.set_ylim(0, max_size_y)
ax_crop.set_xlabel("X")
ax_crop.set_ylabel("Y")
ax_crop.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{path_to_save_figure}4_etap_crop_heatmap.png", dpi=300)
plt.close(fig_crop)
fig_weed, ax_weed = plt.subplots(figsize=(10, 5.5))
sc_weed = ax_weed.scatter(X, Y, c=Z_weed, cmap='Reds', s=300, marker='s')
ax_weed.set_title("Weed Heatmap")
fig_weed.colorbar(sc_weed, ax=ax_weed)
ax_weed.set_xlim(0, max_size_x)
ax_weed.set_ylim(0, max_size_y)
ax_weed.set_xlabel("X")
ax_weed.set_ylabel("Y")
ax_weed.set_aspect('equal')
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}4_etap_weed_heatmap.png", dpi=300)
plt.close(fig_weed)