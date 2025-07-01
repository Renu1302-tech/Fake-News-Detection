# 📰 Fake News Detection Using Machine Learning

This project implements a fake news classification system using traditional machine learning models. It combines two datasets—`Fake.csv` and `True.csv`—to train classifiers that distinguish between real and fake news articles based on their textual content.

---

## 📁 Files in This Repository

- `Fake News Detection.ipynb` – Jupyter Notebook containing the full workflow
- `README.md` – Project overview and setup instructions
- *(Datasets not included due to size)*

---

## 📊 Dataset

Two CSV files are used:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

Each file includes columns like title, text, and subject. A new `label` column is added:
- `1`: Fake news
- `0`: Real news

> ⚠️ Note: The CSV files are not uploaded due to size restrictions. You can download them from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

---

## 🔍 What This Notebook Does

### 📌 Data Preprocessing
- Combine `Fake.csv` and `True.csv` into a single DataFrame
- Drop unnecessary columns (title, subject, etc.)
- Shuffle and split the data
- Clean the text:
  - Remove punctuation and special characters
  - Lowercasing
  - (Optionally) stopword removal and lemmatization

### 🔧 Feature Engineering
- TF-IDF vectorization is used to convert text to numerical format.

### 🤖 Models Used
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**

### 📈 Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🧠 Best Performing Model

At the end, you compare performance and determine which model performs best for fake news detection based on evaluation metrics.

---

## 📦 Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk`, `re`, `string`
- `scikit-learn`
- `xgboost`

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `True.csv` in the notebook's directory
3. Open `Fake News Detection.ipynb` in Jupyter or Google Colab
4. Run all cells to reproduce the analysis and model training

---

