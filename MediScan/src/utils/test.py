from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score , classification_report , f1_score
from joblib import load
from src.utils.config import PROCESSED_DATA_DIR , MODELS_DIR 
from src.features.encoding import x_test_encoding , y_test_scaling

def test_run():
    print("test started ... .. .")
    # load the model
    model = load(MODELS_DIR / "best_mediscan_model.pkl")
    X_tansformer = load(MODELS_DIR/"X_transformer.pkl")
    y_trasformer = load(r"C:\Users\mayan\gitrepos\MEDISCAN\MediScan\models\Y_transfomer.pkl")
    #load the test data
    test_data = pd.read_csv(PROCESSED_DATA_DIR/"test.csv")

    #split the data
    X_test = test_data.drop(columns= ["Disease"])
    y_test = test_data.Disease

    #features encoding
    x_test_encoded = x_test_encoding(X_test , X_tansformer)
    print("X test encoding is complete sucessfully.")
    y_test_encoded = y_test_scaling(y_test , y_trasformer)
    print("X test encoding is complete sucessfully.")

    #predict
    y_pred = model.predict(x_test_encoded)

    #model evleuation
    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test_encoded, y_pred))
    print("test run completed..")

if __name__ == "__main__":
    test_run()