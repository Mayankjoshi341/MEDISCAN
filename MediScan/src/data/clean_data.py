import pandas as pd
def data_clean(main_df : pd.DataFrame):
    for col in main_df.columns:
        main_df[col] = main_df[col].str.replace("_" , " ")
        main_df[col] = main_df[col].fillna("none")
        main_df[col] = main_df[col].str.strip()
    main_df.columns = main_df.columns.str.lower()
    print("Data cleaning is complete successfully")
    return main_df

