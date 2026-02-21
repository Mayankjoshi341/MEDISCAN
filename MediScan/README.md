🩺 MediScan – AI Powered Disease Prediction System
📌 Overview

MediScan is an AI-powered disease prediction web application that analyzes user-selected symptoms and predicts the most probable diseases using machine learning.

The system provides:

🔍 Top-3 disease predictions with confidence scores

📖 Disease description

⚠ Precautions and recommended actions

🏥 Nearby hospital recommendations (based on user location)

📧 Email health report feature

This project focuses on solving a real-world healthcare support problem using Machine Learning and geolocation services.

🚀 Features

Multi-class disease classification (41 diseases)

Symptom-based prediction using trained ML models

Probability ranking of top 3 diseases

Hospital discovery using Google Places API

Distance calculation (sorted nearest → farthest)

OpenStreetMap visualization using Folium

Email report system

Modular ML pipeline architecture

CLI-based training workflow

🧠 How It Works
1️⃣ Data Processing

Dataset contains symptom combinations mapped to diseases

17 symptom columns converted into 132 unique symptoms

MultiLabelBinarizer transforms symptoms into binary vectors

Disease labels encoded using LabelEncoder

Example transformation:

["itching", "skin rash"]  
→ [0,0,1,0,1,0,0,...]
2️⃣ Model Training

Models Used:

Logistic Regression

Random Forest

(Optional: XGBoost)

Hyperparameter tuning done using:

RandomizedSearchCV

Best model selected based on cross-validation accuracy.

Final model and transformers saved using:

joblib
3️⃣ Prediction Flow

When user selects symptoms:

Symptoms → transformed using saved MultiLabelBinarizer

Model predicts probabilities using predict_proba

Top 3 diseases extracted

Results decoded using LabelEncoder

Output:

Disease: Acne
Confidence: 70.66%
4️⃣ Hospital Recommendation System

User location captured via browser geolocation

Google Places API fetches nearby hospitals

Distance calculated using Haversine formula

Hospitals filtered by:

Minimum rating

Minimum number of reviews

Sorted nearest → farthest

Displayed using OpenStreetMap (Folium)

5️⃣ Email Report

User enters Gmail address

Health summary sent via SMTP (App Password authentication)

Includes prediction, precautions, and advice

🏗 Project Structure
MediScan/
│
├── .steamlit/
├── data/
├── models/
├── reports/
├── src/
│   ├── data/
│   ├── features/
│   ├── tuning/
│   ├── modelevaluation/
│   ├── prediction/
│   ├── pipelines/
│   ├── utils/
│   └── app_runner.py
│
├── MediScan_app.py
├── main.py
└── requirements.txt

🖥 Running The Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train Model
python main.py train

3️⃣ Run Streamlit App
streamlit run app.py


📊 Model Performance

41-class classification

Cross-validation accuracy: ~99–100% (due to structured dataset)

Probability-based ranking implemented

Noise sensitivity tested

Robust ranking behavior observed


⚠ Limitations

Dataset contains synthetic and overlapping symptom combinations

Not a medical diagnostic tool

Probabilities represent model confidence, not medical certainty

Real-world medical validation required

🎯 Future Improvements

Probability calibration

Severity-level based triage system

Driving distance integration

Better symptom weighting

Improved dataset with real-world variability

Doctor recommendation system

Deployment on cloud platform

📌 Disclaimer

This system is for educational and research purposes only.
It is not a substitute for professional medical advice.

👨‍💻 Author

Mayank Joshi
Machine Learning & Data Science Enthusiast