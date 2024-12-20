import pandas as pd
import numpy as np
from pandas import DataFrame

def identify_column_types(df, special_feat):

    num_feat = [column for column in df if pd.api.types.is_numeric_dtype(df[column])]
    cat_feat = [column for column in df if not pd.api.types.is_numeric_dtype(df[column])]
    
    for feat in special_feat:
        
        num_feat.remove(feat)

        cat_feat.insert(0,feat)

    # SalePrice is not a feature but the label
    num_feat.remove("SalePrice")
    # Id is not a feature
    num_feat.remove("Id")

    return num_feat, cat_feat

def submission_file(predictions):
    
    if isinstance(predictions, np.ndarray):
        
        pred_list = predictions.flatten().tolist()

    elif isinstance(predictions, pd.DataFrame):

        pred_list = predictions.values.flatten().tolist()


    id_list = list(range(1461, 1461 + len(pred_list)))
    df = pd.DataFrame({
    'Id': id_list,
    'SalePrice': pred_list
    })

    df.to_csv('submission.csv', index=False)
    
    print("List saved to 'submission.csv'")

def align_columns(df_train_prep, df_test_prep):
    """
    Align columns in df_test_prep to match those in df_train_prep.
    Missing columns will be added with zero values.
    
    Parameters:
    - df_train_prep: DataFrame, training data with reference columns.
    - df_test_prep: DataFrame, test data to be aligned.

    Returns:
    - df_test_aligned: DataFrame with columns aligned to df_train_prep.
    """
    # Get the list of columns in df_train_prep
    train_columns = df_train_prep.columns
    
    # Ensure all columns from df_train_prep are in df_test_prep
    for col in train_columns:
        if col not in df_test_prep.columns:
            # Add missing column with zeros
            df_test_prep[col] = 0
    
    # Keep only columns that are in df_train_prep
    df_test_aligned = df_test_prep[train_columns]
    
    return df_test_aligned


def set_column_type(df):

    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Try converting to numeric (float)
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # If conversion fails, ensure it remains as string
            df[col] = df[col].astype(str)
    
    return df

