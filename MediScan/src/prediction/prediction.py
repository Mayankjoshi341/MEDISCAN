import pandas as pd
import numpy as np
def predicted_disease(input_encoded , model , encoder):
    pred_output = model.predict(input_encoded)[0]
    decoded_pred = encoder.inverse_transform([pred_output])
    return decoded_pred[0] , pred_output

def top_3_predictions(input_encoded , model , encoder):
    probs = model.predict_proba(input_encoded)
    top3_indices = np.argsort(probs[0])[::-1][:3]
    top3_probs = probs[0][top3_indices]
    top3_diseases = model.classes_[top3_indices]
    top3_results = []
    for disease, prob in zip(top3_diseases, top3_probs):
        disease_name = encoder.inverse_transform([disease])[0]
        top3_results.append((disease_name, prob))
    return top3_results

def disease_description(info_df : pd.DataFrame ,  disease_name):
    description = info_df[info_df["Disease"] == disease_name]["Description"].values[0]
    if len(description) > 0:
        return description
    else:
        return "Description not available."
def disease_precautions(precautions_df , disease_name):
    precautions = precautions_df[precautions_df["Disease"] == disease_name]
    precautions_list = []
    for index , row in precautions.iterrows():
        precautions_list.append(row['Precaution_1'])
        precautions_list.append(row['Precaution_2'])
        precautions_list.append(row['Precaution_3'])
    return precautions_list
def home_remedies(homecare_df : pd.DataFrame , disease_name):
    disease_row = homecare_df[homecare_df["disease_name"] == disease_name]
    remedies = []
    for index , row in disease_row.iterrows():
        remedies.append(row["home_remedy_1"])
        remedies.append(row["home_remedy_2"])
        remedies.append(row["home_remedy_3"])
    severity_level = disease_row["severity_level"].values[0]
    reaction_advice = disease_row["reaction_advice"].values[0]

    return remedies , severity_level , reaction_advice