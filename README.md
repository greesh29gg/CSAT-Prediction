# 📊 CSAT Score Forecasting Using ML

This project focuses on predicting **Customer Satisfaction (CSAT) Scores** using historical customer support data with machine learning. By analyzing various features such as response time, product type, and support details, we aim to forecast customer satisfaction levels and help businesses enhance their service quality.

---

## 🚀 Project Objective

The main goal is to build an end-to-end machine learning pipeline that:
- Cleans and prepares customer support data
- Engineers meaningful features like response time
- Handles missing values and class imbalance
- Selects top features for prediction
- Trains and evaluates multiple ML models
- Identifies the best model based on performance metrics

---

## 🗂️ Dataset Description

The dataset includes the following key fields:
- `issue_responded` – When the support team responded
- `Issue_reported at` – When the customer reported the issue
- `order_date_time` – Order placement timestamp
- `Survey_response_Date` – When the feedback was given
- `CSAT Score` – Target variable (1 to 5)
- Categorical and numerical features related to support interactions

---

## 🔧 Technologies & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

---

## 🧹 Data Wrangling & Preprocessing

- Converted datetime columns
- Engineered a new feature: `Response Time`
- Handled missing values using median imputation
- Label encoded categorical columns
- Capped outliers using the IQR method
- Adjusted target values for classification

---

## 🧠 Feature Selection

- Used `SelectKBest` with ANOVA F-value to choose the top 10 features
- Removed irrelevant date columns before modeling

---

## ⚖️ Class Imbalance Handling

- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset and ensure fair model learning

---

## 🤖 Models Used

| Model                | Description                   |
|---------------------|-------------------------------|
| Logistic Regression | Baseline linear classifier     |
| Random Forest       | Ensemble model for accuracy    |
| XGBoost             | Gradient boosting model        |

- Hyperparameter tuning was done using `RandomizedSearchCV` on XGBoost to improve performance.

---

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score
- Feature Importance (for interpretability)

---

## 🏆 Results

| Model                | Accuracy     |
|---------------------|--------------|
| Logistic Regression | ~63%         |
| Random Forest       | ~72%         |
| XGBoost (Base)      | ~75%         |
| XGBoost (Tuned)     | **~80%**     |

> ✅ **Tuned XGBoost** was selected as the best model based on accuracy and classification report.

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to predict customer satisfaction effectively. The model helps businesses:
- Detect potential dissatisfaction early
- Improve response time and support quality
- Make data-driven service decisions

---

## 📁 Folder Structure

