import os
import pandas as pd
from django.core.cache import cache

def read_excel_sheets(excel_file):
    """ Read all sheets from an Excel file """
    with pd.ExcelFile(excel_file) as xls:
        left_xdata_df = pd.read_excel(xls, sheet_name="left_xdata")
        left_ydata_df = pd.read_excel(xls, sheet_name="left_ydata")
        right_xdata_df = pd.read_excel(xls, sheet_name="right_xdata")
        right_ydata_df = pd.read_excel(xls, sheet_name="right_ydata")
    return left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df

def load_all_excel_files(normalized_folder):
    """ Load all Excel files from the compare_data directory and cache them """
    normalized_files_data = cache.get('normalized_files_data')

    if normalized_files_data is None:
        normalized_files_data = {}
        for root, _, files in os.walk(normalized_folder):
            for filename in files:
                if filename.endswith('.xlsx'):
                    excel_file = os.path.join(root, filename)
                    try:
                        left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df = read_excel_sheets(excel_file)
                        relative_path = os.path.relpath(excel_file, start=normalized_folder)
                        normalized_files_data[relative_path] = (left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue

        cache.set('normalized_files_data', normalized_files_data, timeout=None)  # Cache data indefinitely

    return normalized_files_data
