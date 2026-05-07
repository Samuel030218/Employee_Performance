# Employee Performance Prediction App using Streamlit
import streamlit as st
import pandas as pd
import joblib

# Load the saved stacked model
# Corrected filename to match the saved model
model = joblib.load("stacking_rf_gbc_model.pkl")

st.title("HR Employee Performance Prediction App")

st.write("""
This app helps HR managers predict employee performance ratings based on various factors.
You can either input employee details manually or upload a CSV file for batch predictions.
""")

# --- Manual Input Section ---
st.header("Manual Input for Employee Performance Prediction")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=65, value=30)
distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=40, value=10)
emp_hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=2)
emp_last_salary_hike_percent = st.slider("Last Salary Hike Percent", min_value=11, max_value=25, value=15)
total_work_experience_in_years = st.number_input("Total Work Experience (Years)", min_value=0, max_value=40, value=10)
training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
experience_years_at_this_company = st.number_input("Experience Years At This Company", min_value=0, max_value=40, value=5)
experience_years_in_current_role = st.number_input("Experience Years In Current Role", min_value=0, max_value=18, value=3)
years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, value=3)

# Mapped categorical inputs
gender_map = {"Male": 0, 
              "Female": 1}
gender = gender_map[st.selectbox("Gender", list(gender_map.keys()))]

education_background_map = {
    "Life Sciences": 1,
    "Medical": 2,
    "Marketing": 3,
    "Technical Degree": 4,
    "Human Resources": 5,
    "Other": 6
}
education_background = education_background_map[
    st.selectbox("Education Background", list(education_background_map.keys()))
]

marital_status_map = {"Married": 1, 
                      "Single": 2, 
                      "Divorced": 3}
marital_status = marital_status_map[
    st.selectbox("Marital Status", list(marital_status_map.keys()))
]

emp_department_map = {
    "Sales": 1, 
    "Development": 2, 
    "Research & Development": 3,
    "Human Resources": 4, 
    "Finance": 5, 
    "Data Science": 6,
    "Operations": 7, 
    "Customer Service": 8, 
    "Quality Assurance": 9,
    "Legal": 10, 
    "Product Management": 11, 
    "IT": 12, 
    "Administration": 13
}
emp_department = emp_department_map[
    st.selectbox("Employee Department", list(emp_department_map.keys()))
]

# Job Role (example mapping, adjust to match training)
emp_job_role_map = {
    "Sales Executive": 1, 
    "Developer": 2, 
    "Research Scientist": 3,
    "Sales Representative": 4, 
    "Human Resources": 5, 
    "Manager": 6,
    "Technical Lead": 7, 
    "Data Scientist": 8, 
    "HR Business Partner": 9,
    "Delivery Manager": 10, 
    "Operations Manager": 11, 
    "Finance Analyst": 12,
    "Product Manager": 13, 
    "Marketing Executive": 14, 
    "Recruiter": 15,
    "Business Analyst": 16, 
    "Quality Analyst": 17, 
    "UX Designer": 18, 
    "Accountant": 19
}
emp_job_role = emp_job_role_map[
    st.selectbox("Employee Job Role", list(emp_job_role_map.keys()))
]

business_travel_map = {"Travel_Rarely": 1, "Travel_Frequently": 2, "Non-Travel": 3}
business_travel_frequency = business_travel_map[
    st.selectbox("Business Travel Frequency", list(business_travel_map.keys()))
]

education_levels = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
emp_education_level = st.slider("Employee Education Level", min_value=1, max_value=5, value=3, step=1)

env_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
emp_environment_satisfaction = env_map[
    st.selectbox("Environment Satisfaction", list(env_map.keys()))
]

job_involvement_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
emp_job_involvement = job_involvement_map[
    st.selectbox("Job Involvement", list(job_involvement_map.keys()))
]

job_level_map = {"Entry": 1, "Junior": 2, "Mid-Level": 3, "Manager": 4, "Senior Manager": 5}
emp_job_level = job_level_map[
    st.selectbox("Job Level", list(job_level_map.keys()))
]

job_satisfaction_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
emp_job_satisfaction = job_satisfaction_map[
    st.selectbox("Job Satisfaction", list(job_satisfaction_map.keys()))
]

relationship_satisfaction_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
emp_relationship_satisfaction = relationship_satisfaction_map[
    st.selectbox("Relationship Satisfaction", list(relationship_satisfaction_map.keys()))
]

work_life_balance_map = {"Bad": 1, "Average": 2, "Good": 3, "Best": 4}
emp_work_life_balance = work_life_balance_map[
    st.selectbox("Work Life Balance", list(work_life_balance_map.keys()))
]

over_time_map = {"No": 0, "Yes": 1}
over_time = over_time_map[
    st.selectbox("Over Time", list(over_time_map.keys()))
]

attrition_map = {"No": 0, "Yes": 1}
attrition = attrition_map[
    st.selectbox("Attrition (Left company)", list(attrition_map.keys()))
]

# Calculate engineered features
tenure_ratio = experience_years_at_this_company / (total_work_experience_in_years + 1)
promotion_gap = years_since_last_promotion / (total_work_experience_in_years + 1)

if st.button("Predict for Single Employee"):
    # Assemble input data into a DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "EducationBackground": [education_background],
        "MaritalStatus": [marital_status],
        "EmpDepartment": [emp_department],
        "EmpJobRole": [emp_job_role],
        "BusinessTravelFrequency": [business_travel_frequency],
        "DistanceFromHome": [distance_from_home],
        "EmpEducationLevel": [emp_education_level],
        "EmpEnvironmentSatisfaction": [emp_environment_satisfaction],
        "EmpHourlyRate": [emp_hourly_rate],
        "EmpJobInvolvement": [emp_job_involvement],
        "EmpJobLevel": [emp_job_level],
        "EmpJobSatisfaction": [emp_job_satisfaction],
        "NumCompaniesWorked": [num_companies_worked],
        "OverTime": [over_time],
        "EmpLastSalaryHikePercent": [emp_last_salary_hike_percent],
        "EmpRelationshipSatisfaction": [emp_relationship_satisfaction],
        "TotalWorkExperienceInYears": [total_work_experience_in_years],
        "TrainingTimesLastYear": [training_times_last_year],
        "EmpWorkLifeBalance": [emp_work_life_balance],
        "ExperienceYearsAtThisCompany": [experience_years_at_this_company],
        "ExperienceYearsInCurrentRole": [experience_years_in_current_role],
        "YearsSinceLastPromotion": [years_since_last_promotion],
        "YearsWithCurrManager": [years_with_curr_manager],
        "Attrition": [attrition],
        "TenureRatio": [tenure_ratio],
        "PromotionGap": [promotion_gap]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Adjust prediction back to original scale if needed
    predicted_rating = int(prediction[0] + 2)  # assuming min PerformanceRating is 2
    st.success(f"Predicted Employee Performance Rating: {predicted_rating}")


# --- Batch Upload Section ---
st.header("Batch Predictions via CSV")

# Define required columns for the model
required_columns = [
    "EmpNumber",
    "PerformanceRating",
    "ExperienceYearsAtThisCompany",
    "TotalWorkExperienceInYears",
    "YearsSinceLastPromotion",
    "EmpEnvironmentSatisfaction",
    "EmpJobInvolvement",
    "EmpJobLevel",
    "EmpJobSatisfaction",
    "EmpRelationshipSatisfaction",
    "EmpWorkLifeBalance",
    "OverTime",
    "Attrition"
]

# Provide a downloadable CSV template
import io

template_df = pd.DataFrame(columns=required_columns)
csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download Empty CSV Template",
    data=csv_buffer.getvalue(),
    file_name="employee_template.csv",
    mime="text/csv"
)

# Provide a downloadable example CSV with sample rows
example_data = pd.DataFrame([
    {
        "EmpNumber": 101,
        "PerformanceRating": 3,
        "ExperienceYearsAtThisCompany": 5,
        "TotalWorkExperienceInYears": 10,
        "YearsSinceLastPromotion": 2,
        "EmpEnvironmentSatisfaction": "High",
        "EmpJobInvolvement": "Medium",
        "EmpJobLevel": "Junior",
        "EmpJobSatisfaction": "High",
        "EmpRelationshipSatisfaction": "Very High",
        "EmpWorkLifeBalance": "Good",
        "OverTime": "Yes",
        "Attrition": "No"
    },
    {
        "EmpNumber": 102,
        "PerformanceRating": 4,
        "ExperienceYearsAtThisCompany": 8,
        "TotalWorkExperienceInYears": 15,
        "YearsSinceLastPromotion": 3,
        "EmpEnvironmentSatisfaction": "Medium",
        "EmpJobInvolvement": "High",
        "EmpJobLevel": "Manager",
        "EmpJobSatisfaction": "Very High",
        "EmpRelationshipSatisfaction": "High",
        "EmpWorkLifeBalance": "Best",
        "OverTime": "No",
        "Attrition": "Yes"
    }
])

example_buffer = io.StringIO()
example_data.to_csv(example_buffer, index=False)

st.download_button(
    label="Download Example CSV with Sample Data",
    data=example_buffer.getvalue(),
    file_name="employee_example.csv",
    mime="text/csv"
)

# File uploader
uploaded_file = st.file_uploader("Upload Employee CSV for Batch Prediction", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Show preview and column info
    st.write("Preview of uploaded data:", data.head())
    st.write(f"Uploaded file has **{len(data.columns)}** columns.")
    st.write("Column names:", list(data.columns))
    
    # Validate required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        st.success("✅ All required columns are present!")

        # --- Define mappings and expected values ---
        mappings = {
            "EmpEnvironmentSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
            "EmpJobInvolvement": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
            "EmpJobLevel": {"Entry": 1, "Junior": 2, "Mid-Level": 3, "Manager": 4, "Senior Manager": 5},
            "EmpJobSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
            "EmpRelationshipSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
            "EmpWorkLifeBalance": {"Bad": 1, "Average": 2, "Good": 3, "Best": 4},
            "OverTime": {"No": 0, "Yes": 1},
            "Attrition": {"No": 0, "Yes": 1}
        }

        # Validate categorical values
        for col, mapping in mappings.items():
            if col in data.columns:
                invalid_values = set(data[col].unique()) - set(mapping.keys())
                if invalid_values:
                    st.warning(f"⚠️ Column '{col}' contains unexpected values: {invalid_values}. Expected: {list(mapping.keys())}")
                data[col] = data[col].map(mapping).fillna(data[col])

        # Feature engineering
        data["TenureRatio"] = data["ExperienceYearsAtThisCompany"] / (data["TotalWorkExperienceInYears"] + 1)
        data["PromotionGap"] = data["YearsSinceLastPromotion"] / (data["TotalWorkExperienceInYears"] + 1)

        # Predictions
        predictions = model.predict(data.drop(columns=['EmpNumber', 'PerformanceRating'], errors='ignore'))

        # Adjust predictions back to original scale if necessary
        data["Predicted_PerformanceRating"] = predictions + 2  # Assuming min PerformanceRating is 2
        st.write("Predictions:", data)

    
# --- Ngrok Setup ---
from pyngrok import ngrok

def start_ngrok(port=8501, auth_token=None):
    """
    Start an ngrok tunnel for the given port.
    Ensures ngrok is only started once.
    """
    # Kill any previous tunnels
    ngrok.kill()

    if auth_token:
        ngrok.set_auth_token(auth_token)

    # Create tunnel
    public_url = ngrok.connect(port)
    return public_url

# Example usage:
# ngrok_auth_token = "YOUR_REAL_NGROK_TOKEN"  # Replace with your token
ngrok_auth_token = st.text_input("Enter your ngrok auth token (optional)", type="password")
public_url = start_ngrok(port=8501, auth_token=ngrok_auth_token)
st.write("Streamlit App Tunnel URL:", public_url)


    
