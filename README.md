# 🦌 Wildlife Poacher Detection and Alerting System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![YOLOv3](https://img.shields.io/badge/YOLOv3-Object%20Detection-red.svg)](https://pjreddie.com/darknet/yolo/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)  

A **real-time wildlife poacher detection and alerting system** built using **YOLOv3 (Deep Learning)**.  
This project uses **custom datasets** and enables monitoring through webcams/external devices to help prevent illegal poaching activities.  

---

## 📖 Table of Contents
1. [Overview](#-overview)  
2. [Dataset](#-dataset)  
3. [Project Structure](#-project-structure)  
4. [Requirements](#-requirements)  
5. [Installation](#-installation)  
6. [Training](#-training)  
7. [Detection](#-detection)  
8. [Sample Results](#-sample-results)  
9. [Resources](#-resources)  
10. [Google Colab Tips](#-google-colab-tips)  
11. [Contributing](#-contributing)  
12. [Contact](#-contact)  

---

## 📖 Overview
- Detects **poachers in real-time** using a webcam or external device.  
- Built on **YOLOv3** trained with a **custom dataset**.  
- Complete workflow: dataset creation → annotation → training → detection → real-time alerts.  
- Can be extended for use in **wildlife sanctuaries, national parks, and forest surveillance**.  

---

## 📂 Dataset
- Collected from **Google Images** using the [Download All Images](https://chrome.google.com/webstore/detail/download-all-images) Chrome extension.  
- Annotated with **[LabelImg](https://github.com/heartexlabs/labelImg)** and **CVAT**.  
- Some pre-labeled datasets are available via [Google Open Images v5](https://storage.googleapis.com/openimages/web/index.html).  

---

## 📂 Project Structure
wildlife-poaching-detection/
│── data/ # Dataset, annotations, obj.data, obj.names
│── cfg/ # Custom YOLOv3 config files
│── backup/ # Trained weights saved here
│── poacher-Implementation.ipynb (private)
│── sample_images/ # Input images
│── outputs/ # Detection outputs
│── README.md


---

## ⚙️ Requirements
- [Darknet](https://github.com/pjreddie/darknet) framework  
- Custom dataset with annotations  
- Config files:  
  - `yolov3-custom.cfg`  
  - `obj.data`  
  - `obj.names`  
- Training file: `train.txt` (and optionally `test.txt`)  
- Pretrained weights:  
  - [YOLOv3](https://pjreddie.com/media/files/yolov3.weights)  
  - [YOLOv3-Tiny](https://pjreddie.com/media/files/yolov3-tiny.weights)  

---
