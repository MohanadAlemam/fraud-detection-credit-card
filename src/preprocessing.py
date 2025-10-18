# Initial required libraries
import numpy as np
import pandas as pd

#01. Class Balance Function
def class_balance(labeled_data : pd.DataFrame):
    """
    Calculate the data class imbalance across the two classes.

    :param labeled_data: Pandas DataFrame containing the data
    :return: data frame containing % of each class in the data and count.
    """
    value_count = labeled_data["Class"].value_counts()
    imbalance_scores = labeled_data["Class"].value_counts()/len(labeled_data) * 100
    # Count each class instances and divide by the total number of instances
    imbalance_scores = imbalance_scores.round(3)

    imbalance_scores = imbalance_scores.rename('Percentage of the Class')
    df = pd.concat([value_count, imbalance_scores], axis=1)

    return df


# 3. Assess Duplication
def check_duplicates(data_frame):
    """
    Checks if the data frame contains duplicate rows.

    :param data_frame: Pandas DataFrame containing the data
    :return: a table with count of duplicate rows class wise, and the total number of duplicate rows.
    """
    duplicated_rows = data_frame.duplicated()
    # record duplicated rows

    duplicates_df = data_frame[duplicated_rows]
    count = len(duplicates_df)
    print(f"\nThere are {count} duplicated rows.\n")

    if count > 0:
        if "Class" in duplicates_df.columns:
            class_counts = duplicates_df["Class"].value_counts()
            print(f"Duplicated rows by class:\n")
            return class_counts
    else:
        print("\nNo further duplicates")