#  Breast Cancer Prediction using Artificial Neural Network (ANN)

This project is a high-accuracy **Artificial Neural Network (ANN)** model designed to classify tumors as **Benign (B)** or **Malignant (M)** using the **Breast Cancer Wisconsin (Diagnostic)** dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-99.12%25-green)

##  Project Overview

Early diagnosis is critical in breast cancer treatment. This study utilizes Deep Learning techniques to analyze 30 different cellular features to predict cancer diagnosis with high precision.

* **Dataset:** 569 patient records with 30 numeric features.
* **Link:** [data_link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Model:** Deep Learning based ANN with Dropout regularization.
* **Performance:** Achieved **99.12% accuracy** on the test set.

---

##  Model Performance & Visualizations

Below are the visualizations representing the model's performance during training and testing.

### 1. Confusion Matrix
The distribution of true positive and false negative predictions. The model correctly identified **all 72 benign cases** and missed only **1 out of 42 malignant cases** in the test set.

<img width="530" height="432" alt="__results___18_1" src="https://github.com/user-attachments/assets/9c98b1f7-cb00-439d-af56-9c71016b5ac4" />

### 2. ROC Curve (Receiver Operating Characteristic)
The ROC curve illustrating the diagnostic ability of the binary classifier system.

<img width="567" height="455" alt="__results___18_3" src="https://github.com/user-attachments/assets/5bda7220-c535-4be2-a367-467c14ff89f0" />

---

##  Model Architecture

The model is built using the `Keras Sequential API`. **Dropout layers** are incorporated between dense layers to prevent overfitting.

| Layer Type | Neurons | Activation | Description |
| :--- | :---: | :---: | :--- |
| **Input Layer** | 30 | - | Input Features (Radius, Texture, etc.) |
| **Dense** | 32 | ReLU | Hidden Layer 1 |
| *Dropout* | - | - | 30% Dropout Rate |
| **Dense** | 16 | ReLU | Hidden Layer 2 |
| *Dropout* | - | - | 20% Dropout Rate |
| **Dense** | 8 | ReLU | Hidden Layer 3 |
| *Dropout* | - | - | 10% Dropout Rate |
| **Output Layer** | 1 | Sigmoid | Binary Classification (0 or 1) |

---

##  Results

Evaluation metrics on the test set (114 samples):

* **Test Accuracy:** 99.12%
* **Precision (Malignant):** 1.00
* **Recall (Malignant):** 0.9762
* **F1-Score:** 0.99

---

##  Installation & Usage

To run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mehmetkosee/breast-cancer-ann.git](https://github.com/mehmetkosee/breast-cancer-ann.git)
    cd breast-cancer-ann
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
    ```

3.  **Run the Notebook:**
    Open `breast-cancer-prediction.ipynb` using Jupyter Notebook or Google Colab to train and evaluate the model.

---

###  Author

**Mehmet Köse**
* GitHub: [@mehmetkosee](https://github.com/mehmetkosee)
* LinkedIn: [@mehmeet-k0se](https://www.linkedin.com/in/mehmeet-k0se/)
