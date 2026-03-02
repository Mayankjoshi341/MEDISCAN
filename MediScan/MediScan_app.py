import pandas as pd
import numpy as np
import math
import streamlit as st
from joblib import load

import re , requests
import time , folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components 
from streamlit_js_eval import streamlit_js_eval

import googlemaps

from src.utils.GmailSender import send_health_report_email
from src.utils.config import MODELS_DIR , RAW_DATA_DIR 
from src.data.load_data import load_detail_data
from src.prediction.processing_input import input_sysptoms , transform_input
from src.prediction.prediction import disease_description, disease_precautions , predicted_disease, home_remedies , top_3_predictions

@st.cache_resource
def load_models():
    model = load(MODELS_DIR/"best_mediscan_model.pkl")
    x_transformer = load(MODELS_DIR/"X_transformer.pkl")
    y_transformer = load(MODELS_DIR/"Y_transfomer.pkl")
    return model, x_transformer, y_transformer

model, x_transformer, y_transformer = load_models()

Description_df , precautions_df , homecare_df  = load_detail_data(RAW_DATA_DIR)

st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide")
st.title("🩺 MediScan - Disease Prediction App")
st.write("Select your symptoms and get an AI-powered disease prediction.")

#getting user location 
coords = streamlit_js_eval(
    js_expressions="""
    new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
            pos => resolve({latitude: pos.coords.latitude, longitude: pos.coords.longitude}),
            err => resolve(null)  // return null if denied/error
        );
    })
    """,
    key="get_location"
)
if coords:
    lat = coords['latitude']
    lng = coords['longitude']
else:
    st.warning("Location access denied. Using default location.")
    lat, lng = 28.6139, 77.2090  # Default: Delhi

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(d_lon/2)**2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Multiselect from trained symptom list
selected_symptoms = st.multiselect("Choose your symptoms:", x_transformer.classes_)
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_data = input_sysptoms(selected_symptoms)
        input_vector = transform_input(input_data, x_transformer)

        decoded_pred, prediction = predicted_disease(
            input_vector, model, y_transformer
        )

        top_3_pred = top_3_predictions(
            input_vector, model, y_transformer
        )

        st.session_state["decoded_pred"] = decoded_pred
        st.session_state["top_3_pred"] = top_3_pred


# ================= DISPLAY PREDICTION =================
if "decoded_pred" in st.session_state:

    decoded_pred = st.session_state["decoded_pred"]
    top_3_pred = st.session_state["top_3_pred"]

    st.subheader(f"🔍 Predicted Disease: **{decoded_pred}**")

    try:
        st.subheader("Description")
        Description = disease_description(Description_df, decoded_pred)
        st.write(Description)

        st.subheader("Precautions to keep in mind")
        Precaution_list = disease_precautions(precautions_df, decoded_pred)
        for i, precaution in enumerate(Precaution_list, 1):
            st.write(f"{i}. {precaution}")

        st.subheader("Recommended home remedies")
        remedies, severity_level, reaction_advice = home_remedies(
            homecare_df, decoded_pred
        )
        for i, remedie in enumerate(remedies, 1):
            st.write(f"{i}. {remedie}")

        st.subheader("Severity Level")
        st.write(severity_level)

        st.subheader("Immediate Advice")
        st.write(reaction_advice)

    except:
        st.write("Nothing to show here")

    with st.expander("Disclaimer"):
        st.error(
            "The information provided here is for educational purposes only and not a substitute for professional medical advice."
        )

    # -------- Top 3 ----------
    st.subheader("Top Predictions")
    for disease, prob in top_3_pred:
        st.write(f"🔹 **{disease}** — {prob*100:.2f}%")

    # ================= HOSPITAL LOGIC =================

    disease_to_keywords = {"Drug Reaction": "emergency hospital","Malaria": "infectious disease hospital","Allergy": "allergy specialist hospital","Hypothyroidism": "endocrinology hospital","Psoriasis": "dermatology hospital","GERD": "gastroenterology hospital",
                                   "Chronic cholestasis": "liver hospital", "hepatitis A": "liver hospital", "Osteoarthristis": "orthopedic hospital", "(vertigo) Paroymsal Positional Vertigo": "neurology hospital", "Hypoglycemia": "endocrinology hospital",
                                   "Acne": "skin hospital","Diabetes": "diabetes specialist hospital","Impetigo": "skin infection hospital","Hypertension": "cardiology hospital","Peptic ulcer diseae": "gastroenterology hospital","Dimorphic hemorrhoids(piles)": "gastroenterology hospital",
                                   "Common Cold": "general hospital", "Chicken pox": "infectious disease hospital", "Cervical spondylosis": "orthopedic hospital", "Hyperthyroidism": "endocrinology hospital", "Urinary tract infection": "urology hospital",
                                   "Varicose veins": "vascular surgery hospital","AIDS": "infectious disease hospital","Paralysis (brain hemorrhage)": "neurology hospital","Typhoid": "infectious disease hospital","Hepatitis B": "liver hospital",
                                   "Fungal infection": "dermatology hospital", "Hepatitis C": "liver hospital", "Migraine": "neurology hospital", "Bronchial Asthma": "pulmonology hospital", "Alcoholic hepatitis": "liver hospital",
                                   "Jaundice": "liver hospital",  "Hepatitis E": "liver hospital",  "Dengue": "infectious disease hospital","Hepatitis D": "liver hospital",  "Heart attack": "cardiology hospital",  "Pneumonia": "pulmonology hospital",
                                   "Arthritis": "rheumatology hospital",  "Gastroenteritis": "gastroenterology hospital", "Tuberculosis": "chest hospital"  }

    Keyword = disease_to_keywords.get(decoded_pred, "hospital")

    if st.button("Find Nearby Hospitals"):

        my_key = st.secrets["GOOGLE_MAPS_API_KEY"]
        gmaps = googlemaps.Client(key=my_key)

        hospital_results = gmaps.places_nearby( # type: ignore
            location=(lat, lng),
            keyword=Keyword,
            radius=4000,
            type="hospital"
        )

        hospitals = []

        for hospital in hospital_results["results"]:
            name = hospital.get("name")
            rating = hospital.get("rating")
            address = hospital.get("vicinity")
            latitude = hospital["geometry"]["location"]["lat"]
            longitude = hospital["geometry"]["location"]["lng"]
            total_rating = hospital.get("user_ratings_total", 0)
            place_id = hospital["place_id"]

        hospital_urls = []
        for place_id in enumerate(hospital_results["results"]):
            maps_url = f"https://www.google.com/maps/search/?api=1&query=Google&query_place_id={place_id[1]['place_id']}"
            hospital_urls.append(maps_url)

            if not rating or rating < 3.5 or total_rating < 30:
                continue
            distance = calculate_distance(lat, lng, latitude, longitude)
            
            hospitals.append({
                "name": name,
                "rating": rating,
                "address": address,
                "latitude": latitude,
                "longitude": longitude,
                "distance": distance
            })
        hospitals = sorted(hospitals, key=lambda x: x["distance"])
        st.session_state["hospitals"] = hospitals
    
        # ================= EMAIL FORM =================
    def is_valid_gmail(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
        return re.match(pattern, email) is not None

    with st.form("email_form"):
        receiver_gmail = st.text_input(
            "Enter your Gmail address:",
            placeholder="example@gmail.com"
        )

        submit_email = st.form_submit_button("Send Health Report")

        if submit_email:
            if not receiver_gmail:
                st.error("Please enter your Gmail.")
            elif not is_valid_gmail(receiver_gmail):
                st.error("Invalid Gmail address.")
            else:
                send_health_report_email(
                    receiver_email=receiver_gmail,
                    symptoms=selected_symptoms,
                    disease=decoded_pred,
                    description=Description,
                    precautions=Precaution_list,
                    home_remedy=remedies[0] if remedies else "N/A",
                    severity_level=severity_level,
                    reaction_advice=reaction_advice,
                    hospital_list=hospital_urls[:3],
                    top3_predictions= top_3_pred)               
                st.success("📧 Report sent successfully!")

# ================= DISPLAY HOSPITALS =================
if "hospitals" in st.session_state:

    hospitals = st.session_state["hospitals"]

    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        st.subheader("List Of Nearest Hospitals")
        for h in hospitals:
            st.write(f"**{h['name']}**")
            st.write(
            f"⭐ {h['rating']} | 📍 {h['address']} | "f"📏 {h['distance']:.2f} km away")      
            st.markdown("---")

    with col2:
        st.subheader("Map View")
        m = folium.Map(location=[lat, lng], zoom_start=13)

        folium.Marker(
            [lat, lng],
            tooltip="You are here",
            icon=folium.Icon(color="blue")
        ).add_to(m)

        for h in hospitals:
            folium.Marker(
                [h["latitude"], h["longitude"]],
                tooltip=h["name"],
                popup=h["name"]
            ).add_to(m)

        st_folium(m, width=700, height=500)

    