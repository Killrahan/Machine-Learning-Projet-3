"""
This file is used to compare two different csv files
"""

import pandas as pd


def compare_csv_files(file1, file2):
    try:
        # Read the CSV files into DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Check if the dimensions match
        if df1.shape != df2.shape:
            print("The files have different dimensions.")
            print(f"File1: {df1.shape}, File2: {df2.shape}")
            return

        # Compare the two DataFrames
        comparison = df1.equals(df2)
        if comparison:
            print("The two CSV files are identical.")
        else:
            print("The two CSV files are different.")
            # Find differences
            diff = df1.compare(df2)
            print("\nDifferences found:")
            print(diff)

    except Exception as e:
        print(f"An error occurred: {e}")


file1 = 'test_labels_sorted.csv'
file2 = 'example_submission.csv'
compare_csv_files(file1, file2)
