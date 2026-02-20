from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score , classification_report , f1_score
from joblib import load
from src.utils.config import PROCESSED_DATA_DIR , MODELS_DIR 

def test_run():

    # load the model
    model = load(MODELS_DIR / "best_mediscan.pkl")

    #load the test data
    test_data = pd.read_csv(PROCESSED_DATA_DIR/"test_data.csv")

    #split the data
    X_test = test_data.drop(columns= ["Diease"])
    y_test = test_data

    #predict
    y_pred = model.predict(X_test)

    #model evleuation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    test_run()