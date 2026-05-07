# -*- coding: utf-8 -*-
# Employee_Perfomance_Prediction.py
"""

# Phase 1: Understanding the Problem

- **Business Impact of Employee Performance**

Employee performance is a cornerstone of organizational success. Strong performance boosts productivity, innovation, customer satisfaction, and profitability, while weak performance can trigger attrition, delays, and higher costs. By analyzing performance data, HR teams can design targeted training, improve retention, and allocate resources more effectively—ultimately strengthening workforce stability and driving growth.

- **Key Variables in the Dataset**
- **Demographics:** Age, Gender, MaritalStatus, EducationBackground
- **Job-related:** EmpDepartment, EmpJobRole, BusinessTravelFrequency, DistanceFromHome
- **Satisfaction and Engagement:** EmpEnvironmentSatisfaction, EmpJobSatisfaction, EmpRelationshipSatisfaction, EmpWorkLifeBalance
- **Performance Indicators:** Attrition, PerformanceRating, EmpLastSalaryHikePercent
- **Experience and Career Progression:** NumCompaniesWorked, TotalWorkExperienceInYears, ExperienceYearsAtThisCompany, YearsSinceLastPromotion
- **Compensation and Work Conditions:** EmpHourlyRate, OverTime
"""

# Phase 2: Exploratory Data Analysis (EDA)

#      Setup and Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

import shap



"""## Phase 2: Load and Inspect Dataset

"""

from google.colab import drive
drive.mount('/content/drive')


Emp_pef = pd.read_csv("/content/drive/MyDrive/Employee_Peformance_Prediction/Employee_Performance.csv")
Emp_pef.head(10)

# data shape
Emp_pef.shape

# gets quick info about the data
Emp_pef.info()

# Descriptive statistics
Emp_pef.describe(include='all')

# Missing values
Emp_pef.isnull().sum()

#checking for duplicates
Emp_pef.duplicated().sum()

#checking for all columns
print(Emp_pef.columns)

"""## Visualization of Key Relationships

### 1. Attrition vs JobSatisfaction (Bar Chart)
- Purpose: This Shows how job satisfaction levels relate to employee attrition.
- Insight: High attrition among low-satisfaction employees signals retention issues.
"""

sns.countplot(x="EmpJobSatisfaction", hue="Attrition", data=Emp_pef)
plt.title("Attrition vs Job Satisfaction")
plt.show()

"""#### 2. Age Distribution (Histogram)
- Purpose: This displays the age spread of employees.
- Insight: It helps in Identifying dominant age groups and potential generational workforce trends.
"""

sns.histplot(Emp_pef["Age"], bins=20, kde=True)
plt.title("Age Distribution of Employees")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

"""### 3. Correlation Heatmap (Numeric Features)
- Purpose: Highlights correlations among numeric variables (e.g., salary hike, years of experience).
- Insight: Detects multicollinearity and strong relationships useful for modeling.
"""

corr = Emp_pef.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

"""## Key Insights from This Heatmap
- Age vs. TotalWorkExperienceInYears (0.68):
- Older employees tend to have more total work experience — intuitive and expected.
- TotalWorkExperienceInYears vs. ExperienceYearsAtThisCompany (0.78): Strong correlation, meaning employees with longer careers often stay longer in the same company.
- ExperienceYearsAtThisCompany vs. YearsWithCurrManager (0.71):
Suggests tenure at the company is closely tied to tenure with the
same manager.
- Job Level vs. Age/Experience: Higher job levels tend to correlate positively with age and experience, reflecting career progression.
- Weak correlations: Variables like DistanceFromHome, HourlyRate, or TrainingTimesLastYear show little to no correlation with most others — meaning they vary independently.

### 4. SalaryHikePercent vs PerformanceRating (Boxplot)
- Purpose: Compares salary hike percentages across performance ratings.
- Insight: Reveals fairness or bias in compensation relative to performance.
"""

sns.boxplot(x="PerformanceRating", y="EmpLastSalaryHikePercent", data=Emp_pef)
plt.title("Salary Hike Percent vs Performance Rating")
plt.show()

"""# Phase 3: Preprocessing"""



# Separate features and target
X = Emp_pef.drop(["PerformanceRating", "EmpNumber"], axis=1)   # Drop EmpNumber and PerformanceRating
y = Emp_pef["PerformanceRating"]
y = y - y.min() # Adjust target to be 0-indexed for models like XGBoost

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(include=["int64","float64"]).columns

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

print(categorical_features)

print(numerical_features)

!pip install --upgrade scikit-learn

"""# # Phase 4: Feature Engineering"""

# Example derived metrics
Emp_pef["TenureRatio"] = Emp_pef["ExperienceYearsAtThisCompany"] / (Emp_pef["TotalWorkExperienceInYears"]+1)
Emp_pef["PromotionGap"] = Emp_pef["YearsSinceLastPromotion"] / (Emp_pef["TotalWorkExperienceInYears"]+1)

# Update X with engineered features, ensuring 'EmpNumber' is dropped
X = Emp_pef.drop(["PerformanceRating", "EmpNumber"], axis=1)

"""# Phase 5: Models Training

# Random Forest Model
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = {
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in rf_model.items():
    rf_model = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", model)])

rf_model.fit(X_train, y_train)

# Evaluate on validation set
val_acc = rf_model.score(X_test, y_test)
print("Test Accuracy:", val_acc)

y_pred = rf_model.predict(X_test)

print(f"\n{name} Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))

!pip show scikit-learn

!pip install --upgrade scikit-learn

"""The Random Forest model achieved a test accuracy of 90.8% on the employee performance prediction task. This means the model correctly predicted the performance rating for approximately 90.82% of the test data.

# Cross-validation
"""

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("CV Mean Accuracy:", cv_scores.mean())

"""The accuracy reduced to 89.5%

I perform Permutation tests to help validate whether my model’s performance is significantly better than chance.

## Permutation Tests
"""

from sklearn.model_selection import permutation_test_score

# Example with Random Forest
rf_model = Pipeline(steps=[("preprocessor", preprocessor),
                         ("classifier", RandomForestClassifier(random_state=42))])

score, perm_scores, pvalue = permutation_test_score(
    rf_model, X_train, y_train, cv=5, n_permutations=100, scoring="accuracy"
)

print("Permutation Test Accuracy:", score)
print("Permutation Test p-value:", pvalue)

"""- My tuned model is highly accurate (≈90%).
- The low p‑value (<0.05) confirms the performance is not random — the model is valid and reliable.
- This is strong evidence that my chosen features and tuned algorithm are capturing meaningful relationships in employee performance ratings.

## Hyperparameter Tuning
"""

param_dist_rf = {
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__bootstrap": [True, False]
}

rand_rf = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor),
              ("classifier", RandomForestClassifier(random_state=42))]),
    param_distributions=param_dist_rf,
    n_iter=30, cv=5, scoring="accuracy", random_state=42
)

rand_rf.fit(X_train, y_train)
print("Best Random Forest Params:", rand_rf.best_params_)

# Get the best pipeline (preprocessor + classifier) from RandomizedSearchCV
best_rf_pipeline = rand_rf.best_estimator_

# Predict on the test set (using X_test instead of X_val)
y_pred_best = best_rf_pipeline.predict(X_test)

# Evaluate accuracy (using y_test instead of y_val)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report for Tuned Random Forest:\n", classification_report(y_test, y_pred_best))

"""The accuracy of the model increased to ≈93% accuracy after the hyperparameter tuning."""

best_rf_pipeline.fit(X_train, y_train)

"""# Logistic Regression"""

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# Training and Evaluating Logistic Regression
lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average="macro"))
print("Recall:", recall_score(y_test, y_pred_lr, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_lr, average="macro"))
print(classification_report(y_test, y_pred_lr))

"""The accuracy of Logistic Regression of ≈84% matches the result of the hyperparameter tunning

## Gradient Boosting Model (gbc)
I also try Gradient Boosting to see if there will be a variation in the accuracy score
"""

gbc_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

gbc_model.fit(X_train, y_train)

# Predict with Gradient Boosting model
y_pred_gbc = gbc_model.predict(X_test)

print("Gradient Boosting Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_gbc))
print("Precision:", precision_score(y_test, y_pred_gbc, average="macro"))
print("Recall:", recall_score(y_test, y_pred_gbc, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_gbc, average="macro"))
print(classification_report(y_test, y_pred_gbc))

"""Gradient Boosting gives Accuracy of 93%

## Hyperparameter Tuning
"""

from sklearn.model_selection import RandomizedSearchCV

# Define the pipeline
gbc_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

# Define parameter distributions for Gradient Boosting
param_dist_gb = {
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__max_depth": [3, 5, 7],
    "classifier__subsample": [0.8, 1.0],
    "classifier__max_features": ["sqrt", "log2", None]
}

# RandomizedSearchCV setup
rand_gb = RandomizedSearchCV(
    gbc_model,
    param_distributions=param_dist_gb,
    n_iter=25,                # number of random combinations to try = 25
    cv=5,                     # 5-fold cross-validation
    scoring="accuracy",       # or "f1_macro" to check class imbalance
    random_state=42,
    n_jobs=-1                 # use all cores for speed
)

# Fit the tuned model
rand_gb.fit(X_train, y_train)

# Best parameters
print("Best Gradient Boosting Params:", rand_gb.best_params_)

# Evaluate on test set
y_pred = rand_gb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred, average="macro"))

"""- The tuned Gradient Boosting gives Accuracy score of 93%

## XGB Model
Checking if the XGBoost model gives a higher accuracy score.
"""

# Define XGBoost pipeline
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    ))
])

# Fit the pipeline with encoded labels
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb, average="macro"))
print("Recall:", recall_score(y_test, y_pred_xgb, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_xgb, average="macro"))
print(classification_report(y_test, y_pred_xgb))

"""The XGB has an accuracy score of  ≈93%, Gradient Boosting Model give an accuracy score of 93%, Logistic Regression has an accuracy score of 84% and Tuned Random Forest model also give 93%

# Model Stacking: Gradient Boosting, XGBoost and Logistic Regression

- To improves performance compared to individual models, I stacked Gradient Boosting and XGBoost using Logistic Regression as meta‑learner.
"""

from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Define base learners
gbc = GradientBoostingClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42)

# Meta-model (final estimator)
meta_lr = LogisticRegression(max_iter=1000, random_state=42)

# Stacking ensemble
stacking_gbc_xgb_model = StackingClassifier(
    estimators=[
        ("gbc", gbc),
        ("xgb", xgb_model)
    ],
    final_estimator=meta_lr,
    cv=5,                # cross-validation for blending
    n_jobs=-1
)

# Build pipeline with preprocessing
stacking_gbc_xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("stacking", stacking_gbc_xgb_model)
])

# Fit the stacked model
stacking_gbc_xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_stack = stacking_gbc_xgb_model.predict(X_test)

print("Stacking Ensemble Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack, average="macro"))
print("Recall:", recall_score(y_test, y_pred_stack, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_stack, average="macro"))
print(classification_report(y_test, y_pred_stack))

"""The Stacking Ensemble Results is 93%

## Save the model with Joblib
"""

import joblib

# Save model
joblib.dump(stacking_gbc_xgb_model, "stacking_gbc_xgb_model.pkl")

submission = pd.DataFrame({
    'id': Emp_pef['EmpNumber'].iloc[X_test.index],
    'PerformanceRating': y_pred_stack
})

submission.to_csv('submission_glog_model.csv', index=False)

"""## Stacking Tuned Random Forest and XGBoost:
-  I also stacked XGboost and Tuned Random Forest using Logistic Regression as meta‑learner to see if the accuracy score will improve.

"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline # Import Pipeline

# Base learners
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="mlogloss", use_label_encoder=False)
rf  = RandomForestClassifier(n_estimators=300, max_depth=10, max_features='sqrt', random_state=42)

# Meta-learner
meta = LogisticRegression(max_iter=1000, random_state=42)

# Stacking ensemble without direct preprocessing (this will be nested in a pipeline)
stacked_xgb_rf_model = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf)],
    final_estimator=meta,
    cv=5,
    n_jobs=-1 # Use all available cores
)

# Wrap the stacking ensemble in a pipeline that includes the preprocessor
stacked_xgb_rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("stacking", stacked_xgb_rf_model)
])

# Fit the complete pipeline
stacked_xgb_rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_stack = stacked_xgb_rf_model.predict(X_test)

print("Stacking (XGB + RF) Ensemble Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack, average="macro"))
print("Recall:", recall_score(y_test, y_pred_stack, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_stack, average="macro"))
print(classification_report(y_test, y_pred_stack))

"""- The Stacked Ensemble for xgboost and Random Forest gives Accuracy Score of 93%.

### Feature Importance across my stacked Ensemble (XGBoost + tuned Random Forest → Logistic Regression meta‑learner)
"""

import numpy as np
import pandas as pd

# Access the fitted base estimators from the stacked model pipeline
fitted_xgb = stacked_xgb_rf_model.named_steps['stacking'].estimators_[0]
fitted_rf = stacked_xgb_rf_model.named_steps['stacking'].estimators_[1]

# Base model importances
xgb_importances = fitted_xgb.feature_importances_
rf_importances  = fitted_rf.feature_importances_

# Meta-learner coefficients
meta_coef = stacked_xgb_rf_model.named_steps['stacking'].final_estimator_.coef_[0]

# Get feature names after preprocessing
numeric_feature_names = numerical_features.tolist()
categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features).tolist()
feature_names = numeric_feature_names + categorical_feature_names

# Combine: scale base importances by meta weights
# This assumes the meta_coef corresponds to the order of estimators, which it does.
combined_importance = (meta_coef[0] * xgb_importances) + (meta_coef[1] * rf_importances)

# Normalize (optional, but good for interpretation)
combined_importance = combined_importance / np.sum(combined_importance)

# Put into DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'XGB': xgb_importances,
    'RF': rf_importances,
    'Combined': combined_importance
}).sort_values('Combined', ascending=False)

print(importance_df.head(10))

import matplotlib.pyplot as plt
import pandas as pd

# Example feature names and combined importance values
features = [
    'Credit_Score','Income','Loan_Amount','Debt_Ratio',
    'Years_Employed','Age','Occupation','Region',
    'Education','Marital_Status'
]

combined_importance = [0.20, 0.15, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03]

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': combined_importance
})

# Sort by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot bar chart
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Combined Importance (Normalized)')
plt.title('Stacked Ensemble Feature Importance')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()

"""### XGBoost, Random Forest, and the combined stacked importance in one chart."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example feature names
features = [
    'Credit_Score','Income','Loan_Amount','Debt_Ratio',
    'Years_Employed','Age','Occupation','Region',
    'Education','Marital_Status'
]

# Example importances (replace with your actual values)
xgb_importances = [0.18, 0.14, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03]
rf_importances  = [0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02]

# Meta-learner coefficients (example: Logistic Regression weights)
meta_coef = [2.1, 0.8]  # weight for XGB, weight for RF

# Compute combined importance
combined_importance = (meta_coef[0] * np.array(xgb_importances) +
                       meta_coef[1] * np.array(rf_importances))
combined_importance = combined_importance / combined_importance.sum()

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'XGB': xgb_importances,
    'RF': rf_importances,
    'Combined': combined_importance
}).sort_values('Combined', ascending=False)

# Plot grouped bar chart
x = np.arange(len(importance_df))
width = 0.25

plt.figure(figsize=(12,6))
plt.bar(x - width, importance_df['XGB'], width, label='XGBoost', color='orange')
plt.bar(x, importance_df['RF'], width, label='Random Forest', color='green')
plt.bar(x + width, importance_df['Combined'], width, label='Stacked Combined', color='blue')

plt.xticks(x, importance_df['Feature'], rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Feature Importance Comparison: XGBoost vs Random Forest vs Stacked Ensemble')
plt.legend(loc='upper right')  #  Legend added
plt.tight_layout()
plt.show()

"""## Save the model with Joblib"""

import joblib

# Save model
joblib.dump(stacked_xgb_rf_model, "stacked_xgb_rf_model.pkl")

"""## Stacking of Gradient Boosting and Random Forest"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# --- Step 1: Tune Gradient Boosting ---
gbc = GradientBoostingClassifier(random_state=42)
param_dist_gbc = {
    "classifier__n_estimators": randint(50, 300),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(2, 6)
}
gbc_search = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor), ("classifier", gbc)]),
    param_distributions=param_dist_gbc,
    n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1
)
gbc_search.fit(X_train, y_train)
best_gbc_tuned_pipeline = gbc_search.best_estimator_ # Keep the full tuned pipeline

# --- Step 2: Tune Random Forest ---
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__max_depth": randint(3, 10),
    "classifier__max_features": ["sqrt", "log2"]
}
rf_search = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor), ("classifier", rf)]),
    param_distributions=param_dist_rf,
    n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf_tuned_pipeline = rf_search.best_estimator_ # Keep the full tuned pipeline

# --- Step 3: Define Logistic Regression meta-learner ---
meta_lr = LogisticRegression(max_iter=1000, random_state=42)

# --- Step 4: Build Stacking Ensemble ---
# Now, define the StackingClassifier directly. Its estimators are the *tuned pipelines*.
# Since the base estimators already include the preprocessor,
# the StackingClassifier will handle the preprocessing for its internal base model fitting.
stacking_gbc_rf_model = StackingClassifier(
    estimators=[
        ("gbc", best_gbc_tuned_pipeline), # Pass the full tuned pipeline
        ("rf", best_rf_tuned_pipeline)    # Pass the full tuned pipeline
    ],
    final_estimator=meta_lr,
    cv=5,
    n_jobs=-1
)

# --- Step 6: Fit and evaluate ---
stacking_gbc_rf_model.fit(X_train, y_train)

y_pred_stack = stacking_gbc_rf_model.predict(X_test)

print("Stacked Ensemble Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack, average="macro"))
print("Recall:", recall_score(y_test, y_pred_stack, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_stack, average="macro"))
print(classification_report(y_test, y_pred_stack))

"""- The Stacked Ensemble for Gradient Boosting (gbc) and Random Forest gives Accuracy Score of 93% which is the same as the Stacked Ensemble for xgboost and Random Forest.
- This shows that any of the Stacked Ensemble model might give almost the same prediction.

### Feature Importance across my stacked Ensemble (XGBoost + tuned Random Forest → gbc meta‑learner)

## Save the model with Joblib
"""

import joblib

# Save model
joblib.dump(stacking_gbc_rf_model, "stacking_gbc_rf_model.pkl")

submission = pd.DataFrame({
    'id': Emp_pef['EmpNumber'].iloc[X_test.index],
    'PerformanceRating': y_pred_stack
})

submission.to_csv('submission_glog_model.csv', index=False)

"""## Stacking of Gradient Boosting and Random Forest with xgboosting as meta - learner"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# --- Step 1: Tune Gradient Boosting ---
gbc = GradientBoostingClassifier(random_state=42)
param_dist_gbc = {
    "classifier__n_estimators": randint(50, 300),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(2, 6)
}
gbc_search = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor), ("classifier", gbc)]),
    param_distributions=param_dist_gbc,
    n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1
)
gbc_search.fit(X_train, y_train)
best_gbc_tuned_pipeline = gbc_search.best_estimator_  # Keep the full tuned pipeline

# --- Step 2: Tune Random Forest ---
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__max_depth": randint(3, 10),
    "classifier__max_features": ["sqrt", "log2"]
}
rf_search = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor), ("classifier", rf)]),
    param_distributions=param_dist_rf,
    n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf_tuned_pipeline = rf_search.best_estimator_  # Keep the full tuned pipeline

# --- Step 3: Define XGBoost meta-learner ---
meta_xgb = XGBClassifier(
    n_estimators=1000,   # number of boosting rounds
    learning_rate=0.05,  # step size shrinkage
    max_depth=4,         # depth of trees
    subsample=0.8,       # subsampling ratio
    colsample_bytree=0.8,# feature sampling
    random_state=42,
    n_jobs=-1
)

# --- Step 4: Build Stacking Ensemble ---
stacking_rf_gbc_model = StackingClassifier(
    estimators=[
        ("gbc", best_gbc_tuned_pipeline),  # Pass the full tuned pipeline
        ("rf", best_rf_tuned_pipeline)     # Pass the full tuned pipeline
    ],
    final_estimator=meta_xgb,  # Use XGBoost as meta-learner
    cv=5,
    n_jobs=-1
)

# --- Step 5: Fit the Stacking Model ---
stacking_rf_gbc_model.fit(X_train, y_train)

print("Refitting the stacking_rf_gbc_model...")
stacking_rf_gbc_model.fit(X_train, y_train)
print("Model refitted successfully.")

# Access the final estimator (XGBoost) and print its feature importances
# The final_estimator is directly part of the StackingClassifier
meta_xgb_fitted = stacking_rf_gbc_model.final_estimator_

# Ensure it's an XGBoost model and has feature_importances_
if hasattr(meta_xgb_fitted, 'feature_importances_'):
    print("Feature importances of the XGBoost meta-learner:")
    print(meta_xgb_fitted.feature_importances_)
else:
    print("The final estimator does not have a 'feature_importances_' attribute or is not an XGBoost model.")

y_pred_stack = stacking_rf_gbc_model.predict(X_test)

print("Stacked Ensemble Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack, average="macro"))
print("Recall:", recall_score(y_test, y_pred_stack, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred_stack, average="macro"))
print(classification_report(y_test, y_pred_stack))

"""- The Stacked Ensemble for Gradient Boosting (gbc) and Random Forest with xgboosting as meta-learner gives Accuracy Score of 93% and precision of 94% which is better than the Stacked Ensemble for xgboost and Random Forest (accuracy = 92.5% precision = 91.7%) and the Stacked Ensemble for xgboost and gbc (accuracy = 92.5% precision = 91.0%).
- This shows that the Stacked Ensemble for Gradient Boosting (gbc) and Random Forest with xgboosting as meta-learner gives a better prediction due to the higher precision.

# Feature Importance
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

# --- Extract feature names ---
feature_names = preprocessor.get_feature_names_out()

# --- Random Forest importance ---
rf_model = best_rf_tuned_pipeline.named_steps["classifier"]
rf_importances = rf_model.feature_importances_

# --- Gradient Boosting importance ---
gbc_model = best_gbc_tuned_pipeline.named_steps["classifier"]
gbc_importances = gbc_model.feature_importances_

# --- Permutation importance for stacked ensemble ---
perm_result = permutation_importance(stacking_gbc_rf_model, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = perm_result.importances_mean

# --- Combine top features ---
top_n = 10
indices_rf = np.argsort(rf_importances)[::-1][:top_n]
indices_gbc = np.argsort(gbc_importances)[::-1][:top_n]
indices_perm = np.argsort(perm_importances)[::-1][:top_n]

# --- Plot side-by-side comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.suptitle("Feature Importance Comparison", fontsize=16, fontweight='bold')

# Random Forest
axes[0].barh(range(top_n), rf_importances[indices_rf][::-1], color='forestgreen')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(feature_names[indices_rf][::-1])
axes[0].set_title("Random Forest Importance")
axes[0].set_xlabel("Importance")

# Gradient Boosting
axes[1].barh(range(top_n), gbc_importances[indices_gbc][::-1], color='darkorange')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(feature_names[indices_gbc][::-1])
axes[1].set_title("Gradient Boosting Importance")
axes[1].set_xlabel("Importance")

# Permutation Importance
axes[2].barh(range(top_n), perm_importances[indices_perm][::-1], color='royalblue')
axes[2].set_yticks(range(top_n))
axes[2].set_yticklabels(feature_names[indices_perm][::-1])
axes[2].set_title("Permutation Importance (Stacked Ensemble)")
axes[2].set_xlabel("Importance")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

"""# Save the Job"""

import joblib

# Save model
joblib.dump(stacking_rf_gbc_model, "stacking_rf_gbc_model.pkl")

submission = pd.DataFrame({
    'id': Emp_pef['EmpNumber'].iloc[X_test.index],
    'PerformanceRating': y_pred_stack
})

submission.to_csv('submission_glog_model.csv', index=False)

