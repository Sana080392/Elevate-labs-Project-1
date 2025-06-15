# 🏥 Healthcare Appointment No-Show Prediction

## 📌 Project Overview
Missed medical appointments can cause inefficient use of healthcare resources and delayed patient care. 
This project aims to predict whether a patient will miss their scheduled appointment using machine learning models, 
and provide actionable insights through Power BI to help clinics reduce no-shows.

---

## 📂 Dataset Description
The dataset contains real appointment records, including patient demographics, appointment dates, health conditions, and whether the patient showed up.

**Key Features:**
- Gender, Age
- SMS received
- Scholarship, Hypertension, Diabetes, Alcoholism
- Days waited between scheduling and appointment
- Day of the week

**Target Variable:** `No-show` (Yes/No)

---

## 🛠 Tools Used
- **Python** (Pandas, Scikit-learn, Seaborn, Matplotlib)
- **Power BI** (Interactive dashboard creation)
- **Jupyter Notebook / VS Code**
- **GitHub** (Version control and submission)

---

## 🧠 Model Summary
We trained and evaluated multiple models:
- **Logistic Regression** (Best performance with balanced results)
- **Random Forest** (Higher recall for no-show class but lower overall accuracy)

**Logistic Regression Accuracy:** ~66.8%  
**F1-score (No-Show):** 0.41

Model saved as: `model/model.pkl`

---

## 📊 Power BI Dashboard Overview
- No-show distribution by age, gender, weekday
- Effect of SMS reminders and waiting time
- Insights on chronic disease and scholarship factors
- Slicers for interactivity and filtering

File: `powerbi_dashboard/dashboard.pbix`

---

## 💡 Key Insights & Recommendations
- Patients with longer waiting periods are more likely to miss appointments
- Scholarship beneficiaries and patients with chronic conditions tend to no-show more often
- Mondays and Saturdays had relatively higher no-show rates
- SMS reminders are not equally effective across all age groups

**Recommendations:**
- Shorten wait times or offer early rebooking
- Target high-risk patients with additional reminders or calls
- Schedule high-risk groups early in the day
- Use multi-channel reminders for different age groups

Full list: [`insights/recommendations.md`](./insights/recommendations.md)

---

## 📎 Deliverables
- 📄 `Report.pdf` – Abstract, methodology, results, and conclusion
- 📊 `dashboard.pbix` – Power BI dashboard with all visual insights
- 🤖 `model.pkl` – Trained logistic regression and random forest model
- 📁 Cleaned dataset: `data/clean_no_show_data.csv`
- 📄 Final notebook: `notebooks/final_model_code.ipynb`

---

## 🙌 Author
**Sana Fathima**  
Student Data Science & Healthcare Innovation  
📬 [LinkedIn] https://www.linkedin.com/in/sana-f-429971358? | ✉️ [E-mail] sanafathimag392@gmail.com

---

