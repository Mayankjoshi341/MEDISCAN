import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

import re , requests
import time , folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components 
from streamlit_js_eval import streamlit_js_eval

import googlemaps

from src.utils.GmailSender import Email_sender
from src.utils.config import MODELS_DIR , RAW_DATA_DIR 
from src.data.load_data import load_detail_data
from src.prediction.processing_input import input_sysptoms , transform_input
from src.prediction.prediction import disease_description, disease_precautions , predicted_disease, home_remedies , top_3_predictions
model = load(MODELS_DIR/"best_mediscan_model.pkl")
x_transformer = load(MODELS_DIR/"X_transformer.pkl")
y_transformer = load(MODELS_DIR/"Y_transfomer.pkl")

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
lat = coords['latitude']
log = coords['longitude']
# Multiselect from trained symptom list
selected_symptoms = st.multiselect("Choose your symptoms:", x_transformer.classes_)
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_data = input_sysptoms(selected_symptoms)
        input_vector = transform_input(input_data , x_transformer)
        prediction = predicted_disease(input_vector,model , y_transformer)
        top_3_pred = top_3_predictions(input_vector , model , y_transformer)

        st.subheader(f"🔍 Predicted Disease: **{prediction[0]}**")
        try:
            st.subheader("Description")
            Description = disease_description(Description_df , prediction)
            st.write(Description)
            st.subheader("Precautions to keep in mind")

            Precaution_list = disease_precautions(precautions_df , prediction)
            for i, precaution in enumerate(Precaution_list, 1):
                st.write(f"{i}. {precaution}")
                         
            st.subheader("Recommended home remedies ")
            remedies , severity_level , reaction_advice = home_remedies(homecare_df , prediction)
            for i, remedie in enumerate(Precaution_list, 1):
                st.write(f"{i}. {remedie}")
            st.subheader("severity_level")
            st.write(severity_level)
            
            st.subheader("Reaction advive to do now ")
            st.write(reaction_advice)
        except:
            st.write("Nothing to show here")
        with st.expander("Disclaimer"):
             st.error("The information provided here is for educational purposes only and not a substitute for professional medical advice.")

        
        st.subheader("Top Predictions")

        for disease, prob in top_3_pred:
            st.write(f"🔹 **{disease}** — {prob*100:.2f}%")
        def is_valid_gmail(email):
            pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
            return re.match(pattern, email) is not None


        if st.button("Send Health Report"):

            receiver_gmail = st.text_input("Enter your Gmail address:",
            placeholder="example@gmail.com")

        if st.button("Confirm & Send"):

            if not receiver_gmail:
                st.error("Please enter your Gmail.")

            elif not is_valid_gmail(receiver_gmail):
                st.error("Invalid Gmail address.")

            else:
                try:
                    #Email_sender()
                    st.success("📧 Report sent successfully!")

                except Exception as e:
                    st.error("Failed to send email.")
                    st.write(e)

        #dict to find the hospital according to the dieseas
        disease_to_keywords = {"Drug Reaction": "emergency hospital","Malaria": "infectious disease hospital","Allergy": "allergy specialist hospital","Hypothyroidism": "endocrinology hospital","Psoriasis": "dermatology hospital","GERD": "gastroenterology hospital",
                               "Chronic cholestasis": "liver hospital", "hepatitis A": "liver hospital", "Osteoarthristis": "orthopedic hospital", "(vertigo) Paroymsal Positional Vertigo": "neurology hospital", "Hypoglycemia": "endocrinology hospital",
                               "Acne": "skin hospital","Diabetes": "diabetes specialist hospital","Impetigo": "skin infection hospital","Hypertension": "cardiology hospital","Peptic ulcer diseae": "gastroenterology hospital","Dimorphic hemorrhoids(piles)": "gastroenterology hospital",
                               "Common Cold": "general hospital", "Chicken pox": "infectious disease hospital", "Cervical spondylosis": "orthopedic hospital", "Hyperthyroidism": "endocrinology hospital", "Urinary tract infection": "urology hospital",
                               "Varicose veins": "vascular surgery hospital","AIDS": "infectious disease hospital","Paralysis (brain hemorrhage)": "neurology hospital","Typhoid": "infectious disease hospital","Hepatitis B": "liver hospital",
                               "Fungal infection": "dermatology hospital", "Hepatitis C": "liver hospital", "Migraine": "neurology hospital", "Bronchial Asthma": "pulmonology hospital", "Alcoholic hepatitis": "liver hospital",
                               "Jaundice": "liver hospital",  "Hepatitis E": "liver hospital",  "Dengue": "infectious disease hospital","Hepatitis D": "liver hospital",  "Heart attack": "cardiology hospital",  "Pneumonia": "pulmonology hospital",
                               "Arthritis": "rheumatology hospital",  "Gastroenteritis": "gastroenterology hospital", "Tuberculosis": "chest hospital"  }
        Keyword = disease_to_keywords[prediction[0]]
        
        #seting us the api
        my_key = st.secrets["GOOGLE_MAPS_API_KEY"]
        gmaps =  googlemaps.Client(key = my_key)

        #geting the palces
        hospital_results = gmaps.places_nearby( # type: ignore
                       location=(lat, log),
                       keyword = Keyword,
                       radius=3000,   # set range for better resultes 
                       type="hospital")
        print("api called")
        #displaying the reults
        hospitals = []
        for hospital in hospital_results["results"]:

            name = hospital.get("name")
            rating = hospital.get("rating" , "NA")
            address = hospital.get("vicinity")
            latitude = hospital["geometry"]["location"]["lat"]
            longitude=  hospital["geometry"]["location"]["lng"]
            total_rating = hospital.get("user_ratings_total", 0)

            if rating < 3.5:
                continue
            if total_rating < 30:
                continue
            if "clinic" in name.lower() or "diagnostic" in name.lower() or "pathology" in name.lower():
                continue

            hospitals.append({
                "name": name,
                "rating": rating,
                "total rating" : total_rating,
                "address": address,
                "latitude": latitude,
                "longitude": longitude
            })


        col1 ,col2 = st.columns(spec=[0.4 , 0.6] , gap= "small" ,  vertical_alignment= "center" , border= True)
        with col1:
                st.subheader("List Of Nearest Hospitals")
                for h in hospitals:
                            st.write(f"**{h['name']}**")
                            st.write(f"⭐ {h['rating']} | 📍 {h['address']}")
                            #st.markdown(f"[Open in Maps](https://www.google.com/maps/search/?api=1&query={h['lat']},{h['lng']})", unsafe_allow_html=True)
                            st.markdown("---")
        with col2:
            st.subheader("Map View")
            m = folium.Map(location=[lat,log], zoom_start=13)

            # Add user marker
            folium.Marker(
                [lat, log],
                tooltip="You are here",
                icon=folium.Icon(color="blue", icon="user")
            ).add_to(m)
            
            # Add hospital markers
            for h in hospitals:
                folium.Marker(
                    [h["latitude"], h["longitude"]],
                    tooltip=h["name"],
                    popup=h["name"]  
                ).add_to(m)
            
            # Render map in Streamlit
            map_data = st_folium(m, width=700, height=500)



        
