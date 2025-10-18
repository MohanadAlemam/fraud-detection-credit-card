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
def check_duplicates(data_frame: pd.DataFrame):
    """
    Checks if the data frame contains duplicate rows.

    :param data_frame: Pandas DataFrame containing the data
    :return: a table with count of duplicate rows class wise, and the total number of duplicate rows.
    """
    duplicated_rows = data_frame.duplicated()
    # record duplicated rows

    duplicates_df = data_frame[duplicated_rows]
    count = len(duplicates_df)

    if count > 0:
        if "Class" in duplicates_df.columns:
            class_counts = duplicates_df["Class"].value_counts()
            print(f"\nThere are total of {count} duplicated rows.")
            print(f"\nDuplicated rows by class:")
            return class_counts
    else:
        print("\nNo further duplicates")

#4 Missing-ness
def missingness(data_frame : pd.DataFrame):
    """
    Calculates the percentage of missing values in each column.
    :param data_frame: data frame containing the data
    :return: percentage of missing values in each column, or a message confirming no missing values.
    """
    for column in data_frame:
        existing = sum(data_frame[column].value_counts())/len(data_frame)
        # count the existing values
        if existing < 1.00:
            existing = existing*100
            print(f"% of missing values in column ({column}) is: {100 - existing:.1f}")
            #Display the missing values %
    else:
        print("No missing values")