import pandas as pd
import joblib
from pathlib import Path
from src.data.load_data import load_data
from src.data.clean_data import data_clean
from src.data.split_data import features_split
from src.features.encoding import symptoms_extract , x_encoding , y_scaler , x_test_encoding , y_test_scaling
from src.features.train_test_split import train_test_split_data
from src.tuning.model_tuning import model_selection
from src.tuning.hyperparameter_tuning import tune_logistic_regression , tune_random_forest  , best_hyperparameters
from src.modelevaluation.evaluate import model_evaluate , evaluate_model_graphs
from src.prediction.processing_input import input_sysptoms , transform_input
from src.prediction.prediction import predicted_disease , top_3_predictions , disease_description , disease_precautions , home_remedies 
from src.utils.config import DATA_DIR , RAW_DATA_DIR , PROCESSED_DATA_DIR , MODELS_DIR , REPORTS_DIR
def full_pipeline():
    # load the data
    print("Pipeline started...")
    main_df = load_data(RAW_DATA_DIR)

    # clean the data
    cleaned_df = data_clean(main_df)
    cleaned_df.to_csv(PROCESSED_DATA_DIR/"cleaned_data.csv")
    
    # split the data
    X , y = features_split(cleaned_df)


    # split the data into train and test
    X_train , X_test , y_train , y_test = train_test_split_data(X , y)

    X_train_df = X_train.copy()
    X_train_df["Disease"] = y_train
    X_train_df.to_csv(PROCESSED_DATA_DIR/"train.csv")

    X_test_df = X_test.copy()
    X_test_df["Disease"] = y_test
    X_test_df.to_csv(PROCESSED_DATA_DIR/"test.csv")


    # extract symptoms
    all_symptoms = symptoms_extract(X)

    # X feature encoding
    X_train_encoded, mlb = x_encoding(X_train)
    x_test_encoded = x_test_encoding(X_test, mlb)
    joblib.dump(mlb, MODELS_DIR / "X_transformer.pkl")


    # y feature encoding
    encoder , y_train_scaled = y_scaler(y_train)
    y_test_scaled = y_test_scaling(y_test , encoder)
    joblib.dump(encoder , MODELS_DIR/ "Y_transfomer.pkl")

    # model selection 
    best_model , model , model_scores_dict = model_selection(X_train_encoded , y_train_scaled)
    print(f"best model is {best_model} with score {model_scores_dict[best_model]}")

    # hyperparameter tuning
    lr_model, lr_score, lr_params= tune_logistic_regression(X_train_encoded , y_train_scaled)
    rf_model, rf_score, rf_params = tune_random_forest(X_train_encoded , y_train_scaled)
    print("Type LR:", type(lr_model))
    print("Type RF:", type(rf_model))

    tuned_model = best_hyperparameters(lr_model, lr_score ,rf_model, rf_score)
    
    joblib.dump(tuned_model ,  MODELS_DIR/"best_mediscan_model.pkl")

    # model evaluation
    model_evaluate(tuned_model , x_test_encoded , y_test_scaled)
    evaluate_model_graphs(tuned_model , x_test_encoded , y_test_scaled)
    print("Pipeline completed successfully.")
    
if __name__ == "__main__":
    full_pipeline()
