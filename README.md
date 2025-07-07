# IIITH AI/ML Internship: End-to-End Computer Vision Projects

A repository documenting the projects and skills developed during my Online Internship at iHub(IIITH). This work covers the complete pipeline of modern computer vision, from data processing and custom dataset creation to model training, evaluation, and interpretation using YOLOv8.

---
## 📜 Project Overview

This internship provided a comprehensive, hands-on journey through the entire lifecycle of a machine learning project. The tasks were designed to build practical skills in object detection and segmentation, culminating in the creation of a fully custom object detector.

The core skills demonstrated across these tasks include:
- **End-to-End ML Pipeline Implementation:** Managing projects from raw data to a final, predictive model.
- **Custom Dataset Creation & Annotation:** Building high-quality, labeled datasets for bespoke computer vision tasks.
- **Model Training & Fine-Tuning:** Leveraging state-of-the-art models like YOLOv8 and training them on new data.
- **Critical Performance Analysis:** Moving beyond simple accuracy metrics to deeply understand and interpret model behavior through graphs and visualizations.

---

## 🚀 Key Tasks and Accomplishments

### 🎯 Task 1 & 2: YOLOv8 Setup and Video Processing

- **Objective:** Set up the Ultralytics YOLOv8 environment and apply it to a real-world video processing task.
- **Process:**
  - Configured a Python environment (Conda/venv) with all necessary dependencies (PyTorch, Ultralytics, OpenCV).
  - Wrote a Python script to perform **object detection** and **image segmentation** on individual images and video streams.
  - Utilized **FFmpeg** via Python's `subprocess` module to programmatically extract frames from a video and re-compile annotated frames back into a final output video.
- **Outcome:** A fully automated script that takes a video, processes each frame with a YOLOv8 model, and generates a new video with the detections overlaid. This established a strong foundation in video data manipulation.

---

### 📊 Task 3: Model Training and Results Interpretation

- **Objective:** Train a YOLOv8 model on a public benchmark dataset and learn to interpret the results.
- **Process:**
  - Trained `yolov8n.pt` on the **African Wildlife Dataset** for 15 epochs.
  - Conducted a deep dive into the output files generated in the `runs/` directory.
- **Interpretation Skills Gained:**
  - **Loss & Metrics Curves (`results.png`):** Analyzed training vs. validation loss to identify learning trends and potential overfitting. Tracked the progression of mAP, Precision, and Recall.
  - **Confusion Matrix (`confusion_matrix.png`):** Identified which classes the model confused most (e.g., rhinos vs. elephants) and assessed performance on background detection (False Positives/Negatives).
  - **PR and F1 Curves:** Understood the trade-off between precision and recall and identified optimal confidence thresholds.
- **Outcome:** A solid understanding of how to evaluate a model's performance quantitatively and qualitatively, a crucial skill for iterating and improving any ML model.


---

### 🏛️ Task 4: Building a Custom Monument Detector

- **Objective:** Apply all previously learned skills to create a custom object detector for a unique problem: identifying world monuments. This was the capstone project demonstrating the end-to-end workflow.
- **Process:**
  1.  **Data Collection:** Curated a custom dataset of **32 distinct monument classes** by downloading over 25 high-quality images.
  2.  **Annotation:** Meticulously annotated each image with bounding boxes using **LabelImg**, ensuring the output was in the correct **YOLO `.txt` format**.
  3.  **Dataset Structuring:** Organized all images and labels into the required `train/` and `val/` directory structure.
  4.  **YAML Configuration:** Created a custom `monuments_config.yaml` file to map the dataset paths and class names.
  5.  **Training:** Launched and monitored the training of `yolov8n.pt` on this brand-new, custom dataset.
- **Outcome:**
  - A **fully functional custom object detection model** trained to recognize specific monuments.
  - Successfully debugged and resolved real-world pipeline issues related to file paths, data formats, and YAML configuration.
  - The final model, while trained on a small dataset, proved the viability of the entire pipeline.
---

## 🛠️ Tech Stack & Tools

- **Languages:** Python
- **Core Libraries:** PyTorch, Ultralytics (YOLOv8), OpenCV, Pandas, NumPy
- **Data Tools:** FFmpeg, `yt-dlp` for video processing
- **Annotation:** LabelImg
- **Environment:** Conda, Pip
- **Version Control:** Git, GitHub

---
