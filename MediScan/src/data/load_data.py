import pandas as pd
def load_data(DATA_DIR):
    print("data loaded successfully.")
    main =  pd.read_csv(DATA_DIR / "dataset.csv")
    return main
def load_detail_data(RAW_DATA_DIR):
                description =  pd.read_csv(RAW_DATA_DIR / "symptom_Description.csv")
                precaution =  pd.read_csv(RAW_DATA_DIR / "symptom_precaution.csv")
                homecare =  pd.read_csv(RAW_DATA_DIR / "disease_homecare_data.csv")
                return description , precaution , homecare
        