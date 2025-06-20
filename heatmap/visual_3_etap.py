from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

def open_image(path):
    """Load image and convert to RGB numpy array."""
    image = Image.open(path).convert("RGB")
    return np.array(image)

def load_annotations(xml_path):
    """Load bounding boxes and class labels from Pascal VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes, labels = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(label)
    return boxes, labels

# Settings
image_dir = "../../crop_weed_research_data/images/test/"
save_path = "2_FIGURE/"
image_ids = [14, 392, 847, 1052, 2957, 4147, 4195, 4219, 4170]

# Grid layout
grid_step = 0.2
num_x, num_y = 3, 3
img_width, img_height = 1137.0, 640.0
scale_x = 1.0
scale_y = img_height / img_width
gap = 0.0

canvas_width = num_x * scale_x + (num_x + 1) * gap
canvas_height = num_y * scale_y + (num_y + 1) * gap

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, canvas_width)
ax.set_ylim(0, canvas_height)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Stage 3: Center points of objects")

for j in range(num_y):
    for i in range(num_x):
        idx = j * num_x + i
        img_id = image_ids[idx]
        img_path = f"{image_dir}image_{img_id}.jpg"
        xml_path = f"{image_dir}image_{img_id}.xml"

        image = open_image(img_path)
        boxes, labels = load_annotations(xml_path)

        # Position on canvas
        x0 = gap + (scale_x + gap) * i
        y0 = gap + (scale_y + gap) * j
        x1 = x0 + scale_x
        y1 = y0 + scale_y

        # Show image
        ax.imshow(image, extent=[x0, x1, y0, y1], aspect='auto')

        # Draw center points
        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            center_x = (xmin + xmax) / 2 / img_width * scale_x + x0
            center_y = (img_height - (ymin + ymax) / 2) / img_height * scale_y + y0
            color = "#00FF00" if label == "crop" else "#FF0000"
            ax.plot(center_x, center_y, marker='o', color=color)


num_cols = int(canvas_width / grid_step)
num_rows = int(canvas_height / grid_step)


width, height = grid_step, grid_step
# rect = patches.Rectangle(
#     (x0, y0),        
#     width, height,   
#     linewidth=1,     
#     edgecolor='black', 
#     facecolor='blue',  
#     alpha=0.3          
# )

for j in range(num_rows):
    for i in range(num_cols):
        x0 = grid_step * i
        y0 = grid_step * j

        rect = patches.Rectangle(
            (x0, y0),        
            width, height,   
            linewidth=1,     
            edgecolor='black', 
            facecolor='blue',  
            alpha=0.3          
        )

        ax.add_patch(rect)


plt.tight_layout()
plt.savefig(f"{save_path}3_etap.png", dpi=300)
plt.show()
