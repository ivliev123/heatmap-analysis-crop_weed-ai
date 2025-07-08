# Libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Open an image from a local path
def open_image_local(path_to_image):
    image = Image.open(path_to_image).convert("RGB")
    image_array = np.array(image)
    return image_array

# Load XML annotations
def load_xml_data(xml_path):
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

    return true_boxes, true_classes

# Paths
# path = "../../../crop_weed_research_data/images/test/"
path = "../../../crop_weed_research_data/images/train/"
path_to_save_figure = "2_FIGURE/"
# data = [14, 392, 847, 1052, 2957, 4147, 4195, 4219, 4170]
data = [3795, 3798, 3808, 3812, 3816, 3824, 3869, 3882, 3889]

# Plot settings
fig, ax = plt.subplots(figsize=(12, 8))
num_img_x, num_img_y = 3, 3
# num_img_x, num_img_y = 1, 1
separation = 0.01
img_size = [1137.0, 640.0]  # width, height
img_scale_x = 1 
img_scale_y = 640.0 / 1137.0
max_size_x = num_img_x * img_scale_x + (num_img_x + 1) * separation
max_size_y = num_img_y * img_scale_y + (num_img_y + 1) * separation

ax.set_xlim(0, max_size_x)
ax.set_ylim(0, max_size_y)
ax.set_xlabel('X, m')
ax.set_ylabel('Y, m')
# ax.set_title("Stage 2: Object detection")

for j in range(num_img_y):
    for i in range(num_img_x):
        index = j * num_img_x + i
        image_path = f"{path}image_{data[index]}.jpg"
        xml_path = f"{path}image_{data[index]}.xml"

        image = open_image_local(image_path)
        true_boxes, true_classes = load_xml_data(xml_path)

        # Position and scale for subplot placement
        x0 = separation + (img_scale_x + separation) * i
        x1 = x0 + img_scale_x
        y0 = separation + (img_scale_y + separation) * j
        y1 = y0 + img_scale_y

        ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')

        # Draw bounding boxes
        for box, cls in zip(true_boxes, true_classes):
            xmin, ymin, xmax, ymax = box
            # Normalize coordinates to subplot space
            box_x0 = x0 + xmin / img_size[0] * img_scale_x
            box_x1 = x0 + xmax / img_size[0] * img_scale_x
            box_y0 = y0 + (img_size[1] - ymax) / img_size[1] * img_scale_y
            box_y1 = y0 + (img_size[1] - ymin) / img_size[1] * img_scale_y

            color = "#00FF00" if cls == "crop" else "#FF0000"
            rect = plt.Rectangle((box_x0, box_y0), box_x1 - box_x0, box_y1 - box_y0,
                                 linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(box_x0, box_y0, cls, color=color, fontsize=6, verticalalignment='top')

# Finalize plot
plt.tight_layout()
plt.savefig(f"{path_to_save_figure}2_etap.png", dpi=300)
# plt.show()
