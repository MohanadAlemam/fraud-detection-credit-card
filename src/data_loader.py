# Initial required libraries
import numpy as np
import pandas as pd

# 01. Data Loader Function
def import_data(file_path : str, file_type: str= "csv"):
    """
    Imports a CSV file into a pandas DataFrame.

    :param file_path: Path to the CSV file.
    :param file_type: Type of file to load. Must be 'csv' (default).
    :return: pd.DataFrame: Loaded data
    """
    try:
        if file_type.lower() == "csv":
            raw_data = pd.read_csv(file_path)
        else:
            raise TypeError("File type must be 'csv'")
            # Handles file type errors
        if raw_data.empty:
            raise ValueError("The CSV file is empty.")
            # Handles empty files
        return raw_data

    except Exception as e:
        raise ValueError(f"Failed to import data: {e}")


#02. Class Balance Function
def class_balance(labeled_data : pd.DataFrame):
    labeled_data = labeled_data.drop_duplicates()

    imbalance_scores = labeled_data["Class"].value_counts()/len(labeled_data) * 100
    # Count each class instances and divide by the total number of instances
    imbalance_scores = imbalance_scores.round(2)
    imbalance_scores = imbalance_scores.rename('Percentage of the Class')

    return imbalance_scores