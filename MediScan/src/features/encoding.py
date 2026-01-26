import pandas as pd
def symptoms_extract(X : pd.DataFrame ):
    all_symptoms = set() # to sort all unique symptoms 
    for col in X:
        all_symptoms.update(X[col].unique()) # get the unique symptoms and put it in the all_symtoms 
    all_symptoms.discard("none")
    all_symptoms.discard("Missing value")
    print(len(all_symptoms))
    return all_symptoms

def x_encoding(X : pd.DataFrame):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    symptoms_list = X.apply(lambda row: row.dropna().tolist(), axis=1)
    X_processed = mlb.fit_transform(symptoms_list)
    print("X encoding is complete sucessfully.")
    return X_processed
def y_scaler(y: pd.Series):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_sclaed = encoder.fit_transform(y)
    print("Y feature encoding complete successfully.")
    return encoder , y_sclaed 

"""def transform_with_scaler(y: pd.Series , encoder):
    y_decoder = encoder."""
