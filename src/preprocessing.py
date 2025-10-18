# Initial required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#01. Class Balance Function
def class_balance(labeled_data : pd.DataFrame):
    """
    Calculate the data class imbalance across the two classes.

    :param labeled_data: Pandas DataFrame containing the feature and labels.
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
            print(f"\nDuplicated rows by class are listed in the table:")
            return class_counts
    else:
        print("\nNo further duplicates")

#4. Missing-ness
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

#5. Visualisation
def histograms_vis(data_frame : pd.DataFrame):
    """
    Visualizes histograms of each column/ feature.

    :param data_frame: data frame containing the features
    :return: visualizes histograms of each column/ feature.
    """
    data_frame = data_frame.drop(columns="Class")
    # remove class from the visualisation
    data_frame.hist(figsize = (20, 16), bins = 35, layout=(6, 5))
    # 6 rows, 5 columns
    plt.tight_layout()
    plt.show()

def box_plots(data_frame : pd.DataFrame):
    """
    Plots box plots of each column/ feature class-wise.

    :param data_frame: data frame containing the features.
    :return: box plots of each column/ feature class-wise.
    """
    features = data_frame.drop(columns= "Class").columns
    n_features = len(features)
    n_columns = 4
    n_rows = 8

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 22))
    axes = axes.flatten()

    for i, col in enumerate(features):
        data_frame.boxplot(column=col, by="Class", ax=axes[i])
        axes[i].set_title(f"[{col}]")
        axes[i].set_xlabel("Y")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
       fig.delaxes(axes[j])
       # removes all extras
    plt.suptitle("")
    plt.show()