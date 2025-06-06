# 🧑‍💼 Employee Salary Prediction

This project predicts employee salaries based on **Job Title** and **Years of Experience** using machine learning models like Linear Regression, Random Forest, and Gradient Boosting. It features an interactive **Streamlit web app** with data visualization and model insights.

---

## 📁 Project Structure

```
📆 Employee Salary Prediction/
🔹 Salary_Data.csv                  # Raw dataset (Job Title, Experience, Salary)
🔹 salary_prediction_model.pkl     # Trained ML model
🔹 job_title_encoder.pkl           # LabelEncoder for Job Title
🔹 app.py                          # Streamlit app file
🔹 model_training.py               # Script to train and evaluate models
🔹 README.md                       # Project documentation (this file)
```

---

## 📊 Features

* Predict salary based on:

  * **Job Title**
  * **Years of Experience**
  * **Or both (combined model)**
* Multiple ML Models:

  * Linear Regression
  * Random Forest Regressor
  * Gradient Boosting Regressor
* Model performance metrics (MSE, R²)
* Interactive **dropdown inputs** for predictions
* **Visualizations**:

  * Job Title vs Average Salary
  * Experience vs Salary scatter plot

---

## 🧠 ML Models

| Model                   | Description                         |
| ----------------------- | ----------------------------------- |
| Linear Regression       | Simple baseline model               |
| Random Forest Regressor | Non-linear, ensemble-based model    |
| Gradient Boosting       | High-performance boosting algorithm |

---

## 🚀 How to Run the App

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Employee-Salary-Prediction.git
   cd Employee-Salary-Prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 📝 Dataset

* `Salary_Data.csv` contains:

  * `Job Title` (Categorical)
  * `Years of Experience` (Numerical)
  * `Salary` (Target)

---

## 📈 Visualizations Included

* 📊 Job Title vs Average Salary
* 📉 Years of Experience vs Salary
* 💡 Salary prediction from user input

---

## 💾 Model Training

Model training and evaluation is done in `model_training.py`. It:

* Encodes job titles
* Splits data into train/test
* Trains multiple models
* Evaluates performance
* Saves the best performing model

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📌 License

This project is open source and available under the [MIT License](LICENSE).
