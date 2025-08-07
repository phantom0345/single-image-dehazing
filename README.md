# 🌫️ Single Image Dehazing using a Hybrid Deep Learning Approach

This project implements a robust method for single image dehazing by combining traditional Digital Image Processing (DIP) techniques with a Deep Learning (DL) model. The core of the project is a U-Net architecture trained to predict a transmission map, which is then used to restore a clear image from a hazy input.

---


## 🎯 Project Overview

Atmospheric conditions like haze, fog, or smoke degrade the quality of images by reducing contrast and color fidelity. This project develops a hybrid pipeline that leverages the strengths of both classical image processing and deep learning to restore clear images.

- **Deep Learning Core**: A U-Net model is trained to predict the transmission map from a hazy image.
- **Image Processing Enhancements**: The pipeline incorporates pre-processing and post-processing steps like median filtering, atmospheric light estimation, CLAHE, and sharpening to refine the final output.

---

## 📁 File Descriptions

- `Untitled1 (1).ipynb`: Jupyter Notebook with the complete training pipeline, including data loading, pre-processing, U-Net training, and evaluation.
- `Untitled2.ipynb`: Inference notebook to test the pre-trained model on real-world hazy images.
- `unet_dehaze_2k (1).pth`: Trained PyTorch model file generated from the training notebook.
- `image1.png`: A sample hazy output image in testing.

---

## 📊 Dataset

The model was trained and evaluated on a subset of the **RESIDE Indoor Validation Dataset**.

- **Data Used**: Due to hardware constraints, the training was limited to:
  - 1,000 hazy images
  - 1,000 corresponding transmission maps
  - 100 clear ground truth images

---

## ⚙️ Methodology

### 1. Pre-processing

- **Sorting & Mapping**: Ensures correct pairing between hazy, clear, and transmission map images.
- **Image Preparation**: All images resized to 256×256 resolution.
- **Noise Reduction**: Median blur applied to reduce noise while preserving edges.
- **Atmospheric Light Estimation**: Dark Channel Prior used to estimate atmospheric light (A).

### 2. U-Net Model Training

- **Architecture**: Lightweight U-Net (~31,000 parameters) to predict transmission map t(x).
- **Training**: 
  - 10 epochs 
  - Adam optimizer
  - Mean Squared Error (MSE) loss

### 3. Restoration & Post-processing

- **Dehazing Equation**:  
  `J(x) = (I(x) - A) / max(t(x), 0.1) + A`  
  Where:  
  - `J(x)` = dehazed image  
  - `I(x)` = hazy input  
  - `t(x)` = predicted transmission map  
  - `A` = estimated atmospheric light  

- **Post-processing Enhancements**:
  - Bilateral Filtering
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Image Sharpening

---

## 📈 Results

- **PSNR**: 16.82 dB
- **SSIM**: 0.625

Despite limited training data, the model shows significant improvements in contrast, color, and structure.

> 📷 *Refer to the last pages of the project report for visual comparisons between hazy and dehazed images.*

---

## 💻 Hardware & Software Requirements

- **RAM**: 8 GB or higher  
- **Processor**: Intel i5 (8th Gen) or equivalent  
- **GPU**: (Optional) NVIDIA GPU with CUDA support  
- **Python**: 3.9+  
- **Libraries**:
  - PyTorch  
  - NumPy  
  - OpenCV  
  - Scikit-learn  
  - Matplotlib  
  - Scikit-image  
  - Pillow  

Install all dependencies:
```bash
pip install torch numpy opencv-python scikit-learn matplotlib scikit-image pillow
```

---

## 🚀 How to Run

### Clone the Repository
```bash
git clone https://github.com/phantom0345/single-image-dehazing.git
cd single-image-dehazing
```

### Train the Model (Optional)
1. Download the RESIDE ITS Validation Set from the official source.
2. Extract `haze/`, `trans/`, and `clear/` folders to a known location.
3. Open `Untitled1 (1).ipynb`, update the folder paths, and run all cells.
4. A new `.pth` model file will be generated.

### Test with Real-World Images
1. Ensure the pre-trained model file `unet_dehaze_2k (1).pth` is in the working directory.
2. Open `Untitled2.ipynb`.
3. Set the image path (e.g., `image1.png`) and run the notebook cells to view dehazed output.

---

## 👨‍💻 Contributor

This project was submitted in partial fulfillment of the requirements for the Bachelor of Technology degree at **SRM University - AP**.

**M. Yaswanth Sai**  
*Reg No: AP22110011084*  
Under the guidance of **Dr. Kanaparthi Suresh Kumar**

---
