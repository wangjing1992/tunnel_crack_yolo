## 📂 Dataset

The road crack dataset used in this project is open-source and hosted on Roboflow Universe.

👉 **Download Dataset:** **[Crack Vudec Dataset](https://universe.roboflow.com/project-creo0/crack-vudec)**

We express our sincere gratitude to the original creators/uploaders (project-creo0) for open-sourcing this data.

### 🗂️ Recommended Directory Structure

To ensure the provided training and validation scripts work without modification, please download the dataset (in YOLO format) and organize your directory structure as follows:

```text
road_crack_yolo/  (Your Repository Root)
├── cfg/
│   └── datasets/
│       └── road_crack.yaml  <-- Point this YAML to the 'data' folder
├── data/                      <-- Create this folder and paste downloaded data here
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/                  <-- If available
│       ├── images/
│       └── labels/
├── ultralytics/
├── train.py
└── val.py