# Heatmap Analysis for Crop and Weed Detection Using Neural Networks

## 📁 Dataset

The dataset used for training neural networks is available at:  
🔗 [Hugging Face Datasets — crop_weed_research_data](https://huggingface.co/datasets/ivliev123/crop_weed_research_data/tree/main)

---

## 🧠 Trained Models

The trained neural network models can be downloaded from:

- 🔗 [Detection Models — Part 1](https://huggingface.co/ivliev123/crop_weed_detection_models_part_1/tree/main)  
- 🔗 [Detection Models — Part 2](https://huggingface.co/ivliev123/crop_weed_detection_models_part_2/tree/main)

---

## 🏗️ Model Architectures Used

The following neural network architectures were explored in this research:

| Folder         | Architecture                               |
|----------------|--------------------------------------------|
| `research__1`  | SSD MobileNet V1 FPN 640x640               |
| `research__2`  | SSD MobileNet V2 FPNLite 640x640           |
| `research__3`  | SSD ResNet50 V1 FPN 640x640                |
| `research__4`  | SSD ResNet101 V1 FPN 640x640               |
| `research__5`  | SSD ResNet152 V1 FPN 640x640               |
| `research__6`  | Faster R-CNN ResNet50 V1 640x640           |
| `research__7`  | Faster R-CNN ResNet101 V1 640x640          |
| `research__8`  | Faster R-CNN ResNet152 V1 640x640          |
| `research__9`  | Faster R-CNN Inception ResNet V2 640x640   |
| `research__10` | EfficientDet D0 512x512                    |
| `research__11` | EfficientDet D1 640x640                    |

---

## 📥 Pretrained Models Used

All models were initialized using official pretrained weights from TensorFlow:

- [`research__1`](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz) — SSD MobileNet V1 FPN  
- [`research__2`](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz) — SSD MobileNet V2 FPNLite  
- [`research__3`](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) — SSD ResNet50 V1 FPN  
- [`research__4`](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz) — SSD ResNet101 V1 FPN  
- [`research__5`](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz) — SSD ResNet152 V1 FPN  
- [`research__6`](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz) — Faster R-CNN ResNet50  
- [`research__7`](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz) — Faster R-CNN ResNet101  
- [`research__8`](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz) — Faster R-CNN ResNet152  
- [`research__9`](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz) — Faster R-CNN Inception ResNet V2  
- [`research__10`](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz) — EfficientDet D0  
- [`research__11`](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz) — EfficientDet D1  

---

## 🎯 Project Goal

The objective of this project is to analyze the accuracy of different neural network architectures for generating **informational heatmaps** representing crop emergence and weed infestation across agricultural fields.

---

## ✍️ Author

**@ivliev123**  
Technologies: Python, TensorFlow 2.x, Object Detection API  
