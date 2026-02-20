import pandas as pd
import numpy as np
def input_sysptoms(selected_symptoms):
    min_symptoms = 17
    selected_symptoms = selected_symptoms[:min_symptoms]
    selected_symptoms += [np.nan] * (min_symptoms - len(selected_symptoms))
    input_data = pd.DataFrame([selected_symptoms])
    return input_data

def transform_input(input_data , mlb):
    input_data = input_data.apply(lambda row: row.dropna().tolist(), axis=1)
    input_encoded = mlb.transform(input_data)
    return input_encoded

