import pandas as pd
import os

class DataLoader:
    def __init__(self, config):
        self.cache_dir = config.cache_dir
        self.cache_path = os.path.join(self.cache_dir, config.cache_file)
        self.data_path = config.raw_data
        self.timestamp_col = config.timestamp_col
        self.value_col = config.value_col

    def load_data(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Creating cache directory at: {self.cache_dir}")

        if os.path.exists(self.cache_path):
            print(f"Loading data from cache: {self.cache_path}")
            df = pd.read_csv(
                self.cache_path,
                index_col=self.timestamp_col,
                parse_dates=True,
            )
        else:
            print(f"Cache not found. Loading raw data from: {self.data_path}")
            df = pd.read_csv(
                self.data_path,
                usecols=[self.timestamp_col, self.value_col]
            )

            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df.set_index(self.timestamp_col, inplace=True)
            print("Sorting data...")
            df.sort_index(inplace=True)
            
            print(f"Saving data at: {self.cache_path}")
            df.to_csv(self.cache_path)

        print(df.head()) 
        return df
