# Employee Performance Prediction (EPP)
## Overview
Employee Performance Prediction (EPP) App is a predictive analytics platform that helps organizations understand workforce dynamics and forecast attrition and performance. It empowers HR leaders and managers with actionable insights, using machine learning to transform raw employee data into clear, strategic decisions.
This project provides a Streamlit web application that predicts employee performance ratings based on HR and workplace data. It is designed to help HR managers and analysts make data‑driven decisions about workforce performance.

The app uses a stacked ensemble model (tuned Random Forest + tuned Gradient Boosting with xgboosting as meta‑learner) trained on employee data. 
Predictions can be made for a single employee via manual input or for multiple employees via CSV batch upload.

## Features
Interactive Streamlit UI for HR managers.

Manual input form: Enter employee details and get instant predictions.

Batch upload: Upload a CSV file of employees for bulk predictions.

Downloadable templates: Empty CSV and example CSV provided for easy formatting.

Feature engineering: Includes derived features like TenureRatio and PromotionGap.

Ngrok integration: Easily share the app via a public tunnel.

Deployment ready: Model saved with joblib for fast loading.
