<div align="center">

# 🌸 Iris Flower Classification

**Comparing 7 machine-learning models (each untuned and tuned) to classify Iris species. About 97.8% test accuracy.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

</div>

---

## 📌 Overview

The project predicts the species of an Iris flower (**Setosa**, **Versicolor** or **Virginica**) from four measurements:

| Feature | Unit |
|---|---|
| Sepal length | cm |
| Sepal width | cm |
| Petal length | cm |
| Petal width | cm |

The dataset (`Iris.csv`) has **150 samples**, 50 per species.

## 🔬 What the notebook does

1. **Exploratory data analysis**: distributions, pair plots and correlations with `seaborn` and `matplotlib`
2. **Preprocessing**: label encoding and a train/test split
3. **Model training**: each model is trained untuned, then tuned with hyperparameter search
4. **Evaluation**: precision, recall, accuracy and macro-F1 on both train and test sets, collected into a comparison table

## 🤖 Models compared

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| Support Vector Machine | scikit-learn |
| XGBoost | xgboost |
| Gaussian Naive Bayes | scikit-learn |
| Neural Network (MLP) | scikit-learn |

## 📊 Results

Most models reach **≈ 97.8% test accuracy** (macro-F1 ≈ 0.978). For example:

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Logistic Regression | 98.1% | 97.8% |
| Logistic Regression (tuned) | 99.0% | 97.8% |
| Decision Tree | 100% | 97.8% |

Untuned trees and XGBoost score 100% on the training set, a sign of overfitting. The tuned variants close that gap
while keeping the same test accuracy. The full comparison table is in the notebook.

## 🚀 Run locally

```bash
git clone https://github.com/mayurvij22/Iris-Flower-Classification.git
cd Iris-Flower-Classification
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook "Iris Flower Classification.ipynb"
```

You can also open `Iris Flower Classification.html` to view the executed notebook without running anything.

## 📁 Project structure

```text
├── Iris Flower Classification.ipynb   # Full analysis and modelling notebook
├── Iris Flower Classification.html    # Executed notebook, viewable in a browser
└── Iris.csv                           # Dataset (150 rows)
```

---

<div align="center">

Made by **[Mayur Patil](https://github.com/mayurvij22)**

</div>
