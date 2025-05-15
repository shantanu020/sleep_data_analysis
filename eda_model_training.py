# eda_model_training.py
import joblib
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Drop Sleep Disorder (too many missing values)
df = df.drop(columns=['Sleep Disorder'])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
columns_to_encode = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']
for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

# Create High Stress label
df['High_Stress'] = (df['Stress Level'] > df['Stress Level'].median()).astype(int)

# Features and Target
X = df[['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']]
y = df['High_Stress']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as file:
    joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl successfully!")
