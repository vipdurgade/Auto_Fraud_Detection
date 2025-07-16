# 🚨 Car Insurance Fraud Detection Using Machine Learning

## 📌 Project Overview

Insurance fraud is a major challenge for insurance companies, costing billions of dollars annually. The goal of this project is to build a robust machine learning model that can automatically detect fraudulent insurance claims with high accuracy. Early detection not only helps insurers save money but also ensures honest claimants are served faster.

---

## 🎯 Objectives

* Identify key patterns and indicators of fraud in insurance claims
* Clean and transform raw claim data for effective model training
* Train and evaluate various machine learning models to classify claims as **fraudulent** or **legitimate**
* Determine the most important features contributing to fraud detection
* Develop a deployable pipeline that can be integrated into real-world systems


## 📂 Workflow & Steps Followed

### 1. 🧹 Data Cleaning

* Handled missing values (e.g., in `collision_type`, `property_damage`, and `police_report_available`)
* Replaced unknown entries (`?`) with `NaN` and filled using mode imputation
* Dropped irrelevant date columns
* Extracted useful information (e.g., pin code and street from address)
* Encoded categorical variables using `factorize` and mapping techniques

### 2. 📊 Exploratory Data Analysis (EDA)

* Visualized missing data using `missingno`
* Checked data distribution, class balance, and unique values in categorical columns
* Examined correlations and patterns among variables

### 3. 🔄 Data Transformation

* Combined cleaned categorical and numerical data into a single feature set
* Ensured all features were numerical and ready for modeling
* Split the dataset into **training** and **testing** sets using `train_test_split`

### 4. 🤖 Model Building & Evaluation

Several models were experimented with, including:

* **XGBoost Classifier**: Final model used
* Evaluated performance using `.score()` on training and test sets
* Visualized **feature importance** to understand model decisions

---

## ✅ Final Results

| Metric                | Value  |
| --------------------- | ------ |
| **Training Accuracy** | 0.9643 |
| **Testing Accuracy**  | 0.9393 |

* The XGBoost model achieved **93.9% accuracy** on unseen test data
* Important fraud indicators included:

  * `incident_severity`
  * `insured_occupation`
  * `total_claim_amount`
  * `collision_type`
  * `policy_csl`

---

## 🧰 Tech Stack

* **Python** (pandas, numpy, matplotlib, seaborn, missingno)
* **Scikit-learn** (for preprocessing)
* **XGBoost** (final model)
* **Joblib** (for model serialization)
* **Streamlit** (for app deployment)

---

## 🚀 How to Run

```bash
#Predict Fraud
https://ivfrauddetection.streamlit.app/

# Clone the repository
git clone https://github.com/your-username/insurance-fraud-detection.git
cd insurance-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📝 Conclusion

This project demonstrates a complete pipeline from data wrangling to machine learning modeling, offering insights into fraud detection in the insurance industry. The final model is accurate, interpretable, and ready for deployment.


