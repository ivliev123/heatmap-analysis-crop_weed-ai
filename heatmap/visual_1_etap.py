# Libraries
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

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
    



# Путь к папке с изображениями
# path = "reserch__9/training_demo/test_output/"

# путь к папке с изображениями и файлам разметки
path = "main_research_data/training_demo/images/test/"
# Список изображений и их координат в 3D
data = [
    14,  
    392, 
    847, 

    1052,
    2957,
    4147,

    4195,
    4219,
    4170,
    # Добавь остальные изображения по аналогии
]

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 6))


num_img_x = 3
num_img_y = 3
separetion_size = 0.02
img_scale_x = 1 
img_scale_y = 640.0/1137.0 
max_size_x = num_img_x * img_scale_x + (num_img_x + 1) *  separetion_size
max_size_y = num_img_y * img_scale_y + (num_img_y + 1) *  separetion_size

print(max_size_x)

# основная диаграмма
ax.set_xlim(0, max_size_x)
ax.set_ylim(0, max_size_y)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Этап 1. Имитационная модель поля")

for j in range(3):
    for i in range(3):
        # Open the image from my computer

        # image = open_image_local(path + data[j * num_img_x + i]["path"])

        # image_path = path + "image_" +data[j * num_img_x + i] + ".jpg"

        image_path = f"{path}image_{data[j * num_img_x + i]}.jpg"
        print(image_path)

        image = open_image_local(image_path)

        # Define the position and size parameters
        x0 = separetion_size + (img_scale_x + separetion_size) * i
        x1 = separetion_size + img_scale_x + (img_scale_x + separetion_size) * i
        y0 = separetion_size + (img_scale_y + separetion_size) * j
        y1 = separetion_size + + img_scale_y + (img_scale_y + separetion_size) * j


        # Define the position for the image axes
        ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')

# Display the plot
plt.savefig("1_etap.png", dpi=300)
plt.show()