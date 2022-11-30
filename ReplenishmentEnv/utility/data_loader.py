import os
import pandas as pd
from typing import Dict

class DataLoader(object):
    def __init__(self):
        pass

    def load_as_df(self, file_path: str) -> pd.DataFrame:
        if not os.path.isabs(file_path):
            # Modify relative path to absolute path
            file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "..", file_path)
        if file_path.endswith(".csv"):
            data_df = pd.read_csv(file_path, comment='#')
        elif file_path.endswith(".tsv"):
            data_df = pd.read_csv(file_path, sep="\t", comment='#')
        elif file_path.endswith(".xlsx"):
            data_df = pd.read_excel(file_path, comment='#')
        else:
            raise NotImplementedError
        return data_df
    
    def load_as_list(self, file_path: str) -> list:
        data_df = self.load_as_df(file_path)
        output_list = list(data_df.to_numpy().flatten())
        return output_list

    def load_as_dict(self, file_path: str, key: str) -> Dict[str, Dict]:
        data_df = self.load_as_df(file_path)
        columns = data_df.columns
        data_dict = {}
        for _, row in data_df.iterrows():
            data_dict[row[key]] = {
                column: row[column]
                for column in columns if column != key
            }
        return data_dict
    
    def load_as_matrix(self, file_path: str) -> pd.DataFrame:
        data_df = self.load_as_df(file_path)
        data_df = data_df.set_index(data_df.columns[0])
        return data_df
