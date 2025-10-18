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
def check_duplicate(data_frame):
    duplicated_rows = data_frame.duplicated()
    # record duplicated rows

    count = len(data_frame[duplicated_rows])
    print(f"There are {count} duplicate rows.\n\n")

    for index, is_duplicated in enumerate(duplicated_rows):
        if is_duplicated:
            print(f"Duplicate row detected, at index: {index}")
    else:
        print("\nNo further duplicates")