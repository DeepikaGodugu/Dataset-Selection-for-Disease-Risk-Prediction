# 🩺 HFR-MADM Based Healthcare Disease Prediction System

A Streamlit-based healthcare analytics and disease prediction system that automatically evaluates multiple medical datasets using **Hesitant Fuzzy Rough – Multi Attribute Decision Making (HFR-MADM)** and selects the best dataset for **Logistic Regression–based risk prediction**.


## 📌 Project Overview

This project helps in:
- Comparing multiple healthcare datasets
- Ranking datasets based on quality using HFR-MADM
- Automatically selecting the best dataset
- Training a machine learning model for disease prediction
- Providing an interactive **patient risk prediction interface**


## 🚀 Features

-  Supports **multiple healthcare datasets**
-  HFR-MADM dataset quality ranking
-  Logistic Regression model for prediction
-  Interactive charts and metrics
-  Feature importance visualization
-  Confusion matrix & classification report
-  Individual patient risk prediction form
-  Advanced UI with sidebar navigation


## 🗂️ Datasets Used

- `Breast_cancer_dataset.csv`
- `brain_stroke.csv`
- `diabetes.csv`
- `heartdisease.csv`
- `hypertension_dataset.csv`
- `indian_liver_patient.csv`
- `kidney_disease.csv`

##  Methodology

### 1. Dataset Preprocessing
- Remove missing values
- Remove duplicates
- Encode categorical features using `LabelEncoder`
- Separate features (X) and target (y)

### 2. Dataset Ranking (HFR-MADM)
Each dataset is evaluated using:
- Number of samples
- Class balance
- Number of features
- Data reliability

A weighted score is calculated and datasets are ranked.

### 3. Model Training
- Train/Test split (80/20)
- Feature scaling using `StandardScaler`
- Logistic Regression classifier
- Model evaluation using accuracy, F1-score, confusion matrix

### 4. Risk Prediction
- User inputs patient values
- Model predicts **Low Risk / High Risk**
- Displays prediction confidence

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

## 🌐 Live Demo
Interactive web application deployed using Streamlit Cloud.

👉 https://healthcare-ml-project-gvskcbxdfc37z84spyouwd.streamlit.app/


