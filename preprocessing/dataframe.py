import pandas as pd 
import os
import dask.dataframe as dd

def merge_csv_files():
    csv_files = [file for file in os.listdir("../../data") if file.endswith('.csv')]
    df_list = []

    for file in csv_files:
        file_path = os.path.join("data", file)
        df = dd.read_csv(file_path) 
        df_list.append(df)

    merged_df = dd.concat(df_list, ignore_index=True)
    merged_df = merged_df.compute()
    merged_df.to_parquet("../../parquet_data/iot_dataset.parquet")
    return("Merged dataframe shape: ", merged_df.shape)

if __name__ == "__main__":
    merge_csv_files()