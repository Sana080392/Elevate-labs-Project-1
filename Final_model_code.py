import pandas as pd

# Step 1: Load dataset from your file path
file_path = r"C:\Users\sanaf\Desktop\Task 7\Project 1\EL 1.csv"
df = pd.read_csv(file_path)

# Step 2: Clean column names: lowercase, replace spaces and hyphens with underscores
df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]

# Step 3: Show cleaned column names to verify
print("Cleaned Columns:", df.columns.tolist())

# Step 4: Convert date columns to datetime
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# Step 5: Map 'no_show' column to binary (Yes = 1, No = 0)
df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})

# Step 6: Drop duplicate records
df = df.drop_duplicates()

# Step 7: Remove rows with negative age
df = df[df['age'] >= 0]

# Step 8: Create new columns
df['appointment_dayofweek'] = df['appointmentday'].dt.day_name()
df['days_waiting'] = (df['appointmentday'] - df['scheduledday']).dt.days

# Step 9: Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Step 10: Basic info and preview
print("\nDataset Info:\n")
print(df.info())
print("\nFirst 5 rows:\n")
print(df.head())

# Step 11: Save cleaned data to new CSV
cleaned_path = r"C:\Users\sanaf\Desktop\Task 7\Project 1\cleaned_noshow_data.csv"
df.to_csv(cleaned_path, index=False)

print(f"\n✅ Cleaned dataset saved to: {cleaned_path}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('cleaned_noshow_data.csv')

# ========== 🔍 EXPLORATORY DATA ANALYSIS ==========

# 1. No-show distribution
sns.countplot(x='no_show', data=df)
plt.title('Show vs No-Show')
plt.xticks([0, 1], ['Showed Up', 'No-Show'])
plt.show()

# 2. Age vs No-show
sns.boxplot(x='no_show', y='age', data=df)
plt.title('Age vs No-Show')
plt.show()

# 3. SMS received vs No-show
sns.barplot(x='sms_received', y='no_show', data=df)
plt.title('No-Show Rate by SMS Received')
plt.ylabel('No-Show Rate')
plt.show()

# 4. No-show by day of the week
sns.barplot(x='appointment_dayofweek', y='no_show', data=df,
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.title('No-Show Rate by Day of the Week')
plt.ylabel('No-Show Rate')
plt.show()

# ========== 🛠 FEATURE ENGINEERING ==========

# Feature Engineering
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# One-hot encode appointment day of week
df = pd.get_dummies(df, columns=['appointment_dayofweek'], drop_first=True)

# Feature selection (remove 'handicap' since it's missing)
features = [
    'age', 'gender', 'sms_received', 'scholarship',
    'hipertension', 'diabetes', 'alcoholism',
    'days_waiting'
] + [col for col in df.columns if col.startswith('appointment_dayofweek_')]

# Final dataset
X = df[features]
y = df['no_show']

# Confirmation
print("Selected Features:", X.columns.tolist())
print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train a Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Show', 'No-Show'],
            yticklabels=['Show', 'No-Show'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Show', 'No-Show']))

# 5. Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importance in Decision Tree')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("🔹 Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Show', 'No-Show']))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Show', 'No-Show'], yticklabels=['Show', 'No-Show'])
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("🔹 Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Show', 'No-Show']))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['Show', 'No-Show'], yticklabels=['Show', 'No-Show'])
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Re‑train with more trees & slightly deeper depth
rf_model = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42)
rf_model.fit(X_train, y_train)

# Predict class & probability
df_results = X_test.copy()
df_results['actual_no_show'] = y_test.values
df_results['pred_no_show']   = rf_model.predict(X_test)
df_results['prob_no_show']   = rf_model.predict_proba(X_test)[:, 1]

# Save for Power BI
df_results.to_csv('rf_predictions_for_powerbi.csv', index=False)

print("Saved → rf_predictions_for_powerbi.csv")
print("Accuracy:", accuracy_score(y_test, df_results['pred_no_show']))
print(classification_report(y_test, df_results['pred_no_show'],
                            target_names=['Show','No‑Show']))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# STEP 1: Add RowNumber to original dataframe before splitting
df = df.reset_index(drop=True)
df['RowNumber'] = df.index

# STEP 2: Recreate X and y with RowNumber included
features = ['age', 'gender', 'sms_received', 'scholarship', 'hipertension',
            'diabetes', 'alcoholism', 'days_waiting',
            'appointment_dayofweek_Monday', 'appointment_dayofweek_Saturday',
            'appointment_dayofweek_Thursday', 'appointment_dayofweek_Tuesday',
            'appointment_dayofweek_Wednesday']

X = df[features]
y = df['no_show']  # 0 = Show, 1 = No-Show

# Add RowNumber as a feature to X for tracking
X['RowNumber'] = df['RowNumber']

# STEP 3: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# STEP 4: Train Random Forest with basic tuning
rf_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42
)
rf_model.fit(X_train.drop(columns='RowNumber'), y_train)

# STEP 5: Predict and build results dataframe
df_results = X_test.copy()  # includes RowNumber
df_results['actual_no_show'] = y_test.values
df_results['pred_no_show'] = rf_model.predict(X_test.drop(columns='RowNumber'))
df_results['prob_no_show'] = rf_model.predict_proba(X_test.drop(columns='RowNumber'))[:, 1]

# STEP 6: Export to CSV
df_results.to_csv('rf_predictions_for_powerbiiii.csv', index=False)
df.to_csv('cleaned_noshow_data.csv', index=False)

# STEP 7: Print accuracy and classification report
print("\n✅ Predictions saved to 'rf_predictions_for_powerbi.csv'\n")
print("🔹 Accuracy:", accuracy_score(y_test, df_results['pred_no_show']))
print("\n🔹 Classification Report:\n")
print(classification_report(y_test, df_results['pred_no_show'],
                            target_names=['Show', 'No-Show']))
print("✅ Cleaned dataset saved with RowNumber as 'cleaned_noshow_data.csv'")

import os
import joblib

# Assume your trained model is in a variable named 'model'
# Example: model = LogisticRegression().fit(X_train, y_train)

# Create a folder named 'model' if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
joblib.dump(model, 'model/final_model.pkl')

print("✅ Model saved successfully as 'model/final_model.pkl'")

