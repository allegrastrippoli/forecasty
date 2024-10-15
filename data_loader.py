import pandas as pd
import os

class DataLoader:
    def __init__(
        self, 
        cache_dir="cache", 
        cache_filename="sorted-data.csv", 
        data_path="../data/data.csv",
        timestamp_col="TimeStamp", 
        value_col="Value"
    ):
        """
        :param timestamp_col: Name of the timestamp column in the CSV.
        :param value_col: Name of the value column in the CSV.
        """
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_filename)
        self.data_path = data_path
        self.timestamp_col = timestamp_col
        self.value_col = value_col

    def load_data(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created cache directory at: {self.cache_dir}")

        if os.path.exists(self.cache_path):
            print(f"Loading data from cache: {self.cache_path}")
            df = pd.read_csv(
                self.cache_path,
                index_col=self.timestamp_col,
                parse_dates=True,
            )
            print("Cached data loaded successfully.")
        else:
            print(f"Cache not found. Loading raw data from: {self.data_path}")
            df = pd.read_csv(
                self.data_path,
                usecols=[self.timestamp_col, self.value_col]
            )

            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            
            df.set_index(self.timestamp_col, inplace=True)
            df.sort_index(inplace=True)
            df.to_csv(self.cache_path)

        print(df.head()) 
        return df


