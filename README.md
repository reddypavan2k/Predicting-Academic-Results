# 🎓 Student Performance Predictor

This project uses machine learning models to predict student performance based on various academic and behavioral factors. It supports training multiple models, evaluating their performance, visualizing data distributions, and making real-time predictions based on user input.

## 📁 Dataset

The project uses a dataset named `AI-Data.csv`. This dataset should contain features such as:

- Raised Hands
- Visited Resources
- Discussion
- Student Absence Days
- And other features related to student activity and performance.

**Target variable**: `Class` (usually 'L', 'M', or 'H' for Low, Medium, or High performance)

## 🚀 Features

- Preprocessing including label encoding and feature selection
- Multiple ML models:
  - Decision Tree
  - Random Forest
  - Perceptron
  - Logistic Regression
  - MLP Classifier (Neural Network)
- Classification report for each model
- Visualizations with Seaborn
- Interactive input for prediction

## 🛠 Requirements

Install dependencies using:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
