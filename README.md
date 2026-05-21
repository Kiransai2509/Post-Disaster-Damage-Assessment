# Post-Disaster Damage Assessment

An automated deep learning pipeline designed to assess and classify building damage following natural disasters using high-resolution satellite imagery.

## 📌 Project Overview
Timely and accurate damage assessment is critical for disaster response and recovery. This project implements a machine learning framework that processes pre- and post-disaster satellite images to classify the severity of building damage. The model is trained on the extensive xBD dataset, leveraging hardware acceleration to handle large-scale spatial data efficiently.

## ✨ Key Features
*   **High-Volume Image Processing:** Analyzes and processes over 5,500 high-resolution satellite images.
*   **Automated Classification:** Categorizes building damage levels based on visual features extracted from the dataset.
*   **Optimized Vectorization:** Utilizes NumPy for robust data extraction and vectorization workflows to seamlessly feed complex spatial datasets into the neural network.
*   **Hardware Acceleration:** Configured to run in hardware-accelerated environments using CUDA and Google Colab for optimized neural network training efficiency.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Libraries:** NumPy, PyTorch, Pandas
*   **Environment:** Google Colab, Jupyter Notebook
*   **Hardware:** CUDA-enabled GPU

## 📊 Dataset
This project utilizes the **xBD Dataset**, a large-scale, high-resolution satellite imagery dataset created specifically for advancing building damage assessment techniques. 

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* A CUDA-enabled GPU (or Google Colab environment)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Kiransai2509/post-disaster-damage-assessment.git](https://github.com/Kiransai2509/post-disaster-damage-assessment.git)
   cd post-disaster-damage-assessment
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
