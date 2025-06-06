import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load experience values from CSV
@st.cache_data
def load_experience_values():
    df = pd.read_csv("Salary_Data.csv")
    experience_vals = sorted(df["Years of Experience"].dropna().unique())
    return experience_vals

# --- Job Title Prediction Section ---
st.title("Employee Salary Prediction Model")
st.subheader("Employee Salary Prediction from Job Title")

@st.cache_resource
def load_job_model_and_encoder(): # Renamed to avoid confusion
    model_job = joblib.load("salary_prediction_model.pkl")
    label_encoder_job = joblib.load("job_title_encoder.pkl")
    return model_job, label_encoder_job

model_job_title, job_encoder = load_job_model_and_encoder()

job_titles = job_encoder.classes_
job_title = st.selectbox("Select Job Title", job_titles)

if st.button("Predict Salary from Job Title"):
    encoded_job = job_encoder.transform([job_title])
    predicted_salary_job = model_job_title.predict(np.array(encoded_job).reshape(-1, 1))
    st.success(f"Predicted Salary for '{job_title}': ₹ {predicted_salary_job[0]:,.2f}")


# --- Years of Experience Prediction Section ---
st.subheader("Employee Salary Prediction from Years of Experience")

@st.cache_resource
def load_experience_model_and_scaler(): # Renamed and added scaler
    model_exp = joblib.load("experience_salary_model.pkl")
    # ONLY LOAD SCALER IF YOU USED ONE DURING TRAINING
    try:
        scaler_exp = joblib.load("experience_scaler.pkl") # Assuming you saved your scaler
    except FileNotFoundError:
        scaler_exp = None # Or handle the case where no scaler exists
        st.warning("No 'experience_scaler.pkl' found. Ensure your model was trained without scaling or provide the scaler.")
    return model_exp, scaler_exp

experience_options = load_experience_values()
model_experience, scaler_experience = load_experience_model_and_scaler()

experience = st.selectbox("Select Years of Experience", experience_options)

if st.button("Predict Salary for Experience"): # Changed button label for clarity
    input_features = np.array([[float(experience)]]) # Always good to explicitly convert to float

    if scaler_experience: # Apply scaling if scaler exists
        input_features_scaled = scaler_experience.transform(input_features)
        predicted_salary_exp = model_experience.predict(input_features_scaled)[0]
    else: # Predict without scaling if no scaler
        predicted_salary_exp = model_experience.predict(input_features)[0]

    st.success(f"Estimated Salary: ₹ {predicted_salary_exp:,.2f}")

# --- Predict from Job Title + Experience (Dropdowns) ---

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("Salary_Data.csv")

df = load_data()

# --- Load Model & Encoder ---
@st.cache_resource
def load_model():
    model = joblib.load("best_salary_prediction_model.pkl")
    encoder = joblib.load("best_salary_job_encoder.pkl")
    return model, encoder

model, job_encoder = load_model()

# --- User Input ---
st.subheader("Employee Salary Prediction from Years of Experience and Job Title")

# Dropdown for job title
job_titles = job_encoder.classes_
selected_title = st.selectbox("Select Job Title for prediction", job_titles)

# Dropdown for years of experience
experience_vals = sorted(df["Years of Experience"].dropna().unique())
selected_experience = st.selectbox("Select Years of Experience for prediction", experience_vals)

# --- Prediction ---
if st.button("Predict Salary"):
    encoded_title = job_encoder.transform([selected_title])[0]
    input_features = np.array([[encoded_title, float(selected_experience)]])
    
    prediction = model.predict(input_features)[0]
    st.success(f"Estimated Salary for **{selected_title}** with **{selected_experience} years** of experience: ₹ {prediction:,.2f}")

# --- chart representing the variations for experience ---

st.subheader("Experience Years vs Salary Distribution Chart")


# Scatter plot of Experience_Years vs Salary
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x='Years of Experience', y='Salary', data=df)
#plt.title('Experience Years vs Salary')
#plt.xlabel('Experience Years')
#plt.ylabel('Salary')
#plt.show()

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Years of Experience', y='Salary', data=df, ax=ax)
ax.set_title('Experience Years vs Salary')
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')

# Display plot in Streamlit
st.pyplot(fig)

# --- chart representing the variations for job title ---
#st.subheader("Job title vs Salary Distribution Chart")

# Calculate average salary per job title
#avg_salary_by_job = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False)

# Create bar chart
#fig, ax = plt.subplots(figsize=(12, 6))
#sns.barplot(x=avg_salary_by_job.index, y=avg_salary_by_job.values, palette="viridis", ax=ax)
#ax.set_title("Average Salary by Job Title", fontsize=5)
#ax.set_xlabel("Job Title", fontsize=5)
#ax.set_ylabel("Average Salary", fontsize=5)
#plt.xticks(rotation=45, ha='right')


# Display plot in Streamlit
#st.pyplot(fig)