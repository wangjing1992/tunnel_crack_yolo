# tunnel_crack_yolo
A deep learning project for tunnel crack detection based on YOLOv11 models.

## 📖 Introduction

This repository provides the official PyTorch implementation for tunnel crack detection using our proposed **YOLOv11-FDPN-TADDH (TCD_YOLO)** network. 

To achieve a lightweight and efficient model suitable for real-world deployment, we primarily utilize the **LAMP pruning algorithm** to compress the TCD_YOLO network. This pruning strategy significantly reduces the model's parameters and computational overhead while maintaining high detection accuracy for tunnel cracks.

## 🛠️ Prerequisites & Environment

To ensure compatibility and reproduce the results, we recommend using the following specific environment configuration:

* **Python:** `3.10.19`
* **PyTorch:** `2.1.2+cu118`
* **Base Framework Version:** `ultralytics == 8.3.9`

## 🙏 Acknowledgements

Our codebase is built upon the foundational framework provided by **[Ultralytics](https://github.com/ultralytics/ultralytics)**. We express our gratitude to their team for the outstanding open-source contribution.
