Single Image Dehazing using a Hybrid Deep Learning Approach
This project implements a robust method for single image dehazing by combining traditional Digital Image Processing (DIP) techniques with a Deep Learning (DL) model. The core of the project is a U-Net architecture trained to predict a transmission map, which is then used to restore a clear image from a hazy input.

📜 Table of Contents
Project Overview

File Descriptions

Dataset

Methodology

1. Pre-processing

2. U-Net Model Training

3. Restoration & Post-processing

Results

Hardware & Software Requirements

How to Run

Contributor

🎯 Project Overview
Atmospheric conditions like haze, fog, or smoke degrade the quality of images by reducing contrast and color fidelity. This project develops a hybrid pipeline that leverages the strengths of both classical image processing and deep learning to restore clear images.

Deep Learning Core: A U-Net model is trained to predict the transmission map from a hazy image.

Image Processing Enhancements: The pipeline incorporates pre-processing and post-processing steps like median filtering, atmospheric light estimation, Contrast Limited Adaptive Histogram Equalization (CLAHE), and sharpening to refine the final output.

📁 File Descriptions
Untitled1 (1).ipynb: This Jupyter Notebook contains the complete pipeline for data loading, pre-processing, training the U-Net model on the RESIDE dataset, and evaluating its performance.

Untitled2.ipynb: This notebook is for inference. It loads the pre-trained model and tests its dehazing capabilities on real-world images that are not from the training dataset.

unet_dehaze_2k (1).pth: This is the trained PyTorch model file generated from the U-Net training process in Untitled1 (1).ipynb.

image1.png: An example real-world hazy image used for testing in Untitled2.ipynb.

📊 Dataset
The model was trained and evaluated on a subset of the RESIDE (REalistic Single Image DEhazing) Indoor Validation Dataset.

Dataset Link: RESIDE Indoor Validation Set

Data Used: Due to hardware constraints, the model was trained on a subset of 2,100 images (1,000 Hazy, 1,000 Transmission maps, 100 Clear).

⚙️ Methodology
The dehazing process is broken down into three main stages:

1. Pre-processing
Sorting & Mapping: The dataset files are sorted to ensure correct pairing between hazy, clear, and transmission map images.

Image Preparation: Images are resized to 256x256.

Noise Reduction: A median blur filter is applied to smooth the hazy images.

Atmospheric Light Estimation: The Dark Channel Prior method is used to estimate the atmospheric light A.

2. U-Net Model Training
Architecture: A lightweight U-Net model (~31,000 parameters) is used to predict the transmission map t(x).

Training: The model is trained for 10 epochs using the Adam optimizer and Mean Squared Error (MSE) loss.

3. Restoration & Post-processing
Dehazing Equation: The clear image J(x) is recovered using the atmospheric scattering model:
J(x) = (I(x) - A) / max(t(x), 0.1) + A

Enhancements: The output is refined using Bilateral Filtering, CLAHE, and Sharpening to improve visual quality.

📈 Results
The model achieved promising results on the test set:

Peak Signal-to-Noise Ratio (PSNR): 16.82 dB

Structural Similarity Index (SSIM): 0.625

Visual Results
Here are some examples of the dehazing process from the project report:

Original Hazy

Final Enhanced

<img src="https://www.google.com/search?q=https://i.imgur.com/L6gDZuD.png" width="300"/>

<img src="https://www.google.com/search?q=https://i.imgur.com/xQ8v4g2.png" width="300"/>

<img src="https://www.google.com/search?q=https://i.imgur.com/yO8f8aG.png" width="300"/>

<img src="https://www.google.com/search?q=https://i.imgur.com/f7n6Y5C.png" width="300"/>

<img src="https://www.google.com/search?q=https://i.imgur.com/a9h8aHk.png" width="300"/>

<img src="https://www.google.com/search?q=https://i.imgur.com/a4o3t2F.png" width="300"/>

💻 Hardware & Software Requirements
RAM: 8 GB or higher

Processor: Intel i5 (8th Gen) or equivalent

GPU: (Optional) NVIDIA GPU with CUDA support for faster training.

Python: 3.9+

Key Libraries: PyTorch, NumPy, OpenCV, Scikit-learn, Matplotlib, Scikit-image, Pillow.

Install dependencies using pip:

pip install torch numpy opencv-python scikit-learn matplotlib scikit-image pillow

🚀 How to Run
Clone the repository:

git clone [https://github.com/phantom0345/single-image-dehazing.git](https://github.com/phantom0345/single-image-dehazing.git)
cd single-image-dehazing

Train the Model (Optional):

Download the RESIDE ITS Validation set from the official link.

Extract and place the haze, trans, and clear folders in your desired location.

Open Untitled1 (1).ipynb, update the folder paths, and run the cells to train the model. A new .pth file will be generated.

Test with Real-World Images:

Ensure the pre-trained model unet_dehaze_2k (1).pth is in the same directory.

Open Untitled2.ipynb.

Update the path to your test image (e.g., image1.png).

Run the cells to see the dehazed output.

🧑‍💻 Contributor
This project was submitted in partial fulfillment of the requirements for the Bachelor of Technology degree at SRM University - AP.

M. Yaswanth Sai - AP22110011084

Under the guidance of Dr. Kanaparthi Suresh Kumar.
