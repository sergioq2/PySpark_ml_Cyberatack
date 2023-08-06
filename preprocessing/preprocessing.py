import os
import json
import pandas as pd
import shutil

current_path = os.getcwd()
kaggle_json_path = os.path.join(current_path, "kaggle.json")

with open(kaggle_json_path) as f:
    kaggle_json_data = json.load(f)

os.environ["KAGGLE_USERNAME"] = kaggle_json_data["username"]
os.environ["KAGGLE_KEY"] = kaggle_json_data["key"]


def unzip_files():
    zip_file_path = "nbaiot-dataset.zip"
    destination_folder = "data"
    shutil.unpack_archive(zip_file_path, destination_folder)
    return None

def add_target() -> None:
    for file in os.listdir("../../data"):
        name = file.split(".")[-2]
        dataframe = pd.read_csv("../../data/"+file)
        dataframe['target'] = name
        dataframe.to_csv("../../data/"+file, index=False)
    
def main():
    !kaggle datasets download -d mkashifn/nbaiot-dataset
    unzip_files()
    add_target()
    return None

if __name__ == "__main__":
    main()