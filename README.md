Flipkart CSAT Prediction Project
🔹 Project Overview

This project is a Machine Learning Capstone based on Flipkart’s customer support data. The goal is to predict Customer Satisfaction Score (CSAT) using features such as item price, connected handling time, and service channels.

The workflow includes:

Exploratory Data Analysis (EDA)

Data Preprocessing (cleaning, encoding, scaling)

Model Building (Logistic Regression & Random Forest)

Model Evaluation (Accuracy, Precision, Recall, F1)

Hyperparameter Tuning (GridSearchCV)

Feature Importance Analysis

Saving & Loading Model (for deployment readiness)

🔹 Files in This Repository

Customer_support_data.csv → Raw dataset

Sample_EDA_Submission_Template.ipynb → Exploratory Data Analysis notebook

Sample_ML_Submission_Template.ipynb → Machine Learning model notebook

Flipkart project.pptx → Project presentation slides

best_model.pkl → Saved Random Forest model (for deployment)

🔹 Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn (EDA/visualization)

Scikit-learn (ML models, evaluation, GridSearchCV)

Joblib (model saving/loading)

Jupyter Notebook

🔹 How to Run the Project

Clone the repository:

git clone https://github.com/YourUsername/Flipkart-CSAT-Project.git


Open the notebooks in Jupyter Notebook / Jupyter Lab / VS Code.

Run cells in order (EDA → ML → Save & Load model).

The best-performing model (Random Forest) will be saved as best_model.pkl.

Load the saved model and test with unseen data for predictions.

🔹 Results

Random Forest Classifier achieved higher accuracy than Logistic Regression.

Feature importance analysis showed connected_handling_time and channel_name as major contributors to CSAT.

Final model is saved and ready for deployment.

🔹 Author

Rachana Nagaraj
