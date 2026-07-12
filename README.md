# 🛒 ShopSmart – E-Commerce Purchase Prediction
## 🎯 Problem
Predict whether a customer will complete a purchase based on their browsing and behavioral data on an e-commerce platform. This helps businesses identify high-intent customers and optimize marketing/retention efforts.
## 🛠️ Approach

Performed EDA to check feature distributions, class imbalance, and correlations
Handled missing values
Encoded categorical features using OneHotEncoder/LabelEncoderScaled numerical features using StandardScaler 
 Built preprocessing with ColumnTransformer + Pipeline to prevent data leakage between train/test splits
 Trained a Decision Tree Classifier as a baseline model accuracy of 90.47 %
 Trained a Random Forest Classifier (ensemble of decision trees) to reduce overfitting get accuracy 87.63 %
 Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix.

## ⚙️ Tools & Libraries

Python
pandas, numpy
scikit-learn
matplotlib, seaborn

## 📦 Installation
git clone https://github.com/bhaveshyadav111/MachineLearning.git
cd MachineLearning
pip install -r requirements.txt

## ▶️ Usage
jupyter notebook / Kaggle notebook / googleColab
