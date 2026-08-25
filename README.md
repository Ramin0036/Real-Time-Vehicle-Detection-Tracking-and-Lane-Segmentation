# 🚗 Real-Time Vehicle Detection, Tracking & Lane Segmentation

A real-time computer vision system for **vehicle detection, multi-object tracking, and road lane segmentation**.

This project combines a custom-trained **YOLO26s** object detection model, **BoT-SORT** multi-object tracking, and a deep learning-based **lane segmentation model** into a unified video processing pipeline.

The system can detect vehicles, assign persistent tracking IDs, and simultaneously segment road lane regions in real time.

---

## ✨ Features

- 🚗 Real-time vehicle detection
- 🎯 Custom YOLO26s object detection model
- 🆔 Multi-object tracking with BoT-SORT
- 🔢 Persistent Track IDs
- 🛣️ Road lane segmentation
- 🎥 Real-time video processing
- ⚡ GPU-accelerated inference
- 🔗 Combined detection, tracking, and segmentation pipeline
- 📦 Conversion of custom annotations to YOLO format
- 📊 Model training and evaluation

---

## 🧠 System Overview

The complete system consists of three main computer vision components:

Input Video
↓
YOLO26s Object Detection
↓
BoT-SORT Object Tracking
↓
Lane Segmentation Model
↓
Combined Output

For every video frame:

1. The lane segmentation model predicts the road lane region.
2. YOLO26s detects cars, trucks, and pedestrians.
3. BoT-SORT associates detections across frames.
4. Each tracked object receives a persistent Track ID.
5. Detection, tracking, and segmentation results are combined.
6. The final processed frame is displayed and saved as a video.

---

# 📂 Project Structure

Real-Time-Vehicle-Detection-Tracking-Lane-Segmentation/
│
├── dataset/
│   ├── frames/
│   └── lane-masks/
│
├── yolo_dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   │
│   └── data.yaml
│
├── notebooks/
│   ├── Convert_to_YOLO.ipynb
│   ├── YOLO_CAR_Detection.ipynb
│   ├── Tracking.ipynb
│   └── Combined_model.ipynb
│
├── models/
│   ├── best.pt
│   ├── bestmodel.keras
│   └── osnet_x1_0_msmt17.pt
│
├── runs/
│   └── detect/
│       └── train-3/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── videos/
│   └── input.mp4
│
├── results/
│   └── output.mp4
│
├── requirements.txt
└── README.md

---

# 📊 Datasets

The project uses separate datasets for object detection and lane segmentation.

## 🚗 Vehicle Detection Dataset

The object detection dataset is based on the Udacity Self-Driving Car Dataset.

Dataset source:

https://github.com/udacity/self-driving-car/tree/master/annotations

The original annotations contain bounding boxes using:

- xmin
- ymin
- xmax
- ymax

These annotations were converted into the YOLO annotation format.

---

## 🛣️ Lane Segmentation Datasets

Two lane segmentation datasets were combined to build the segmentation dataset used in this project.

### Dataset 1

https://www.kaggle.com/datasets/rangalamahesh/preprocessed-1

### Dataset 2

https://www.kaggle.com/datasets/mehmetokuyar/line-segmentation

The datasets were combined and prepared as image-mask pairs for semantic segmentation.

---

# 🏷️ Object Detection Classes

The object detection model contains three classes:

| Class | ID |
|---|---:|
| Car | 0 |
| Truck | 1 |
| Pedestrian | 2 |

Class mapping:

- Car → 0
- Truck → 1
- Pedestrian → 2

---

# 🔄 Dataset Conversion to YOLO Format

The original dataset annotations use bounding box coordinates:

xmin, ymin, xmax, ymax

YOLO requires bounding boxes in the following format:

class_id x_center y_center width height

The coordinates are normalized according to the image dimensions.

The conversion is performed using:

x_center = ((xmin + xmax) / 2) / image_width

y_center = ((ymin + ymax) / 2) / image_height

width = (xmax - xmin) / image_width

height = (ymax - ymin) / image_height

The resulting YOLO label files contain normalized values.

Example:

0 0.523438 0.481250 0.154167 0.203125

where:

- 0 = class ID
- 0.523438 = normalized x-center
- 0.481250 = normalized y-center
- 0.154167 = normalized width
- 0.203125 = normalized height

---

# 🗂️ YOLO Dataset Configuration

The generated dataset is configured using data.yaml.

Configuration:

path: ./yolo_dataset/
train: images/train
val: images/val
nc: 3
names:
  - car
  - truck
  - pedestrian

The dataset was divided into:

- 90% Training
- 10% Validation

using a fixed random seed.

---

# 🎯 Object Detection with YOLO26s

The object detector was developed using YOLO26s.

The pretrained YOLO26s model was used as the starting point and fine-tuned on the custom vehicle detection dataset.

Training configuration:

- Model: YOLO26s
- Image Size: 320 × 320
- Epochs: 45
- Batch Size: 32
- Patience: 5
- Optimizer: AdamW
- Workers: 0
- Pretrained: Yes

The best trained model was saved as:

runs/detect/train-3/weights/best.pt

---

# 📈 Object Detection Results

The YOLO26s model was evaluated on the validation dataset.

Validation results:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Car | 0.865 | 0.766 | 0.862 | 0.551 |
| Truck | 0.819 | 0.595 | 0.685 | 0.450 |
| Pedestrian | 0.755 | 0.384 | 0.480 | 0.216 |
| Overall | 0.813 | 0.582 | 0.676 | 0.406 |

Validation dataset:

- Images: 942
- Instances: 7,198

The model achieved its strongest performance on the Car class.

Pedestrian detection was more challenging, mainly reflected in the lower recall and mAP values.

---

# 🆔 Object Tracking with BoT-SORT

After object detection, the detected objects are passed to BoT-SORT for multi-object tracking.

The tracker uses:

- Bounding box coordinates
- Detection confidence
- Object class
- Motion information
- Appearance information when Re-ID is enabled

The tracker is initialized using the BoT-SORT implementation provided by BoxMOT.

The Re-ID model used in the project is:

OSNet x1.0 trained on MSMT17

Weights:

osnet_x1_0_msmt17.pt

---

# 🔢 How Track IDs Are Assigned

For each video frame, YOLO produces detections containing:

- Bounding box
- Confidence score
- Class ID

These detections are passed to BoT-SORT.

Example:

Frame t:

Car → ID 1
Car → ID 2
Truck → ID 3

When the next frame arrives, BoT-SORT compares the new detections with existing tracks.

Frame t+1:

Car → ID 1
Car → ID 2
Truck → ID 3

The goal is to maintain the same identity for the same physical object across consecutive frames.

For example, if a car moves from the left side of the frame to the right side, its Track ID can remain unchanged:

Frame 100:
Car → ID 7

Frame 101:
Car → ID 7

Frame 102:
Car → ID 7

This allows the system to distinguish between multiple vehicles and follow their movement over time.

---

# 🛣️ Lane Segmentation

The second major component of the system is road lane segmentation.

A deep learning segmentation model is used to identify lane regions in every video frame.

The segmentation model is loaded from:

bestmodel.keras

Before inference, each frame is:

1. Converted from BGR to RGB.
2. Resized to 256 × 256.
3. Preprocessed using ResNet50 preprocessing.
4. Passed through the segmentation model.
5. Converted into a segmentation mask.
6. Resized back to the original video resolution.

The segmentation output is then overlaid on the original video frame.

---

# 🎨 Lane Visualization

The predicted segmentation mask is converted into a colored mask and combined with the original frame.

The final visualization contains:

- Original video frame
- Lane segmentation overlay
- Vehicle bounding boxes
- Vehicle Track IDs

The lane mask is blended with the original frame to produce the final output.

---

# 🔗 Combined Detection, Tracking and Segmentation

The final pipeline combines all components into one system.

For every frame:

1. Read the frame from the input video.
2. Run lane segmentation.
3. Run YOLO26s object detection.
4. Extract bounding boxes, confidence scores, and class IDs.
5. Pass detections to BoT-SORT.
6. Receive tracked objects and Track IDs.
7. Draw tracking bounding boxes.
8. Draw Track IDs.
9. Overlay the lane segmentation mask.
10. Save the resulting frame to the output video.
11. Display the processed frame in real time.

Final output:

Vehicle Detection + Vehicle Tracking + Lane Segmentation

---

# ⚙️ Main Models

| Component | Model |
|---|---|
| Object Detection | YOLO26s |
| Object Tracking | BoT-SORT |
| Re-Identification | OSNet x1.0 MSMT17 |
| Lane Segmentation | Custom Deep Learning Segmentation Model |

---

# 💻 Environment

The main training and inference environment used for the project includes:

- Python 3.13.9
- PyTorch 2.13.0+cu130
- Ultralytics 8.4.114
- TensorFlow / Keras
- OpenCV
- NumPy
- BoxMOT
- CUDA GPU acceleration

Hardware used during training:

- NVIDIA GeForce RTX 5060 Laptop GPU
- 8 GB VRAM

---

# 📦 Installation

Clone the repository:

git clone <YOUR-REPOSITORY-URL>

cd <YOUR-REPOSITORY-NAME>

Create a virtual environment:

python -m venv venv

Activate the environment.

Linux / WSL:

source venv/bin/activate

Windows:

venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt

---

# ▶️ Running the Project

After installing the required dependencies and downloading the model weights, configure the paths in the combined inference script.

Important paths include:

YOLO_MODEL_PATH

SEGMENTATION_MODEL_PATH

REID_WEIGHTS_PATH

VIDEO_PATH

Then run the combined pipeline.

The system will:

- Read the input video.
- Detect vehicles.
- Track vehicles.
- Assign Track IDs.
- Segment road lanes.
- Display the processed frames.
- Save the final output video.

Press Q to stop the video processing.

---

# 🎥 Demo

The final system output demonstrates:

- Vehicle detection
- Vehicle tracking
- Persistent Track IDs
- Lane segmentation
- Combined real-time visualization

<img width="288" height="512" alt="output_gif" src="https://github.com/user-attachments/assets/b12dd747-1575-4b8e-9179-16f105ab16c7" />

---

# 📸 Visual Results

The final output contains a visualization similar to:

Original Frame
+
Lane Segmentation
+
Vehicle Bounding Boxes
+
Track IDs

Example visualization:

![Detection, Tracking and Lane Segmentation](assets/result.png)

---

# 📊 Performance Summary

The object detection component achieved:

- Overall Precision: 0.813
- Overall Recall: 0.582
- mAP@50: 0.676
- mAP@50-95: 0.406

The best individual detection performance was obtained for cars:

- Precision: 0.865
- Recall: 0.766
- mAP@50: 0.862
- mAP@50-95: 0.551

The complete system combines these detection results with real-time tracking and lane segmentation to create a unified road-scene analysis pipeline.

---

# 🚀 Future Improvements

Possible improvements for future versions include:

- Increase YOLO input resolution.
- Train larger YOLO models such as YOLO26m or YOLO26l.
- Increase the number of training epochs.
- Improve pedestrian detection.
- Improve truck detection.
- Use a stronger Re-ID model.
- Improve lane segmentation accuracy.
- Add vehicle counting.
- Add lane-based vehicle counting.
- Estimate vehicle speed.
- Detect lane departures.
- Add traffic violation detection.
- Optimize the pipeline for higher FPS.
- Export models to TensorRT or other optimized inference formats.
- Add real-time FPS monitoring.

---

# 🔬 Limitations

The current system has several limitations:

- Pedestrian detection performance is lower than car detection performance.
- Lane segmentation accuracy depends on lighting and road conditions.
- Tracking performance can degrade during heavy occlusion.
- Small objects may be difficult to detect at 320 × 320 resolution.
- Running TensorFlow segmentation and YOLO inference simultaneously can increase computational requirements.
- Performance may vary depending on the input video's resolution and frame rate.

---

# 🧩 Technologies Used

- Python
- OpenCV
- NumPy
- TensorFlow
- Keras
- PyTorch
- Ultralytics YOLO
- YOLO26s
- BoxMOT
- BoT-SORT
- OSNet
- Scikit-learn
- Pandas
- Matplotlib
- PIL

---

# 📚 References

## YOLO / Ultralytics

https://docs.ultralytics.com/

## BoxMOT

https://github.com/mikel-brostrom/boxmot

## BoT-SORT

https://github.com/NirAharon/BoT-SORT

## Udacity Self-Driving Car Dataset

https://github.com/udacity/self-driving-car/tree/master/annotations

## Lane Segmentation Dataset 1

https://www.kaggle.com/datasets/rangalamahesh/preprocessed-1

## Lane Segmentation Dataset 2

https://www.kaggle.com/datasets/mehmetokuyar/line-segmentation

---

# 👨‍💻 Author

Developed by **Ramin Allahverdizadeh**

GitHub:

https://github.com/Ramin0036

---

# ⭐ Project Goal

The main goal of this project is to develop an integrated real-time computer vision system capable of understanding important elements of a road scene.

Instead of treating object detection, object tracking, and lane segmentation as independent tasks, this project combines them into a single pipeline:

Vehicle Detection
+
Multi-Object Tracking
+
Lane Segmentation
=
Real-Time Road Scene Understanding

The resulting system provides a foundation for more advanced applications such as autonomous driving assistance, traffic monitoring, vehicle counting, lane departure detection, and intelligent transportation systems.
