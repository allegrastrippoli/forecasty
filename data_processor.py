import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os

class DataProcessor:
    def __init__(self, config):
        self.history_size = config.history_size
        self.target_size = config.target_size
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size  
        self.train_split = config.train_split
        self.step = config.step
        self.model_dir = config.model_dir 

    def scale_data(self, df: pd.DataFrame) -> np.ndarray:
        arr = df.values
        train_split = int(len(arr) * self.train_split)
        train_mean = arr[:train_split].mean(axis=0)
        train_std = arr[:train_split].std(axis=0)
        scaled_arr = (arr - train_mean) / train_std
        scaler = {
            'mean': train_mean,
            'std': train_std
        }

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            print(f"Scaler saved to {scaler_path}")

        return scaled_arr

    def multistep_data(
        self,
        dataset: np.ndarray,
        start_index: int,
        end_index: int | None,
        single_step: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f'Before reshaping: {dataset.shape}')

        data = []
        labels = []

        start_index += self.history_size

        if end_index is None:
            end_index = len(dataset) - self.target_size

        for i in range(start_index, end_index):
            if i + self.target_size > len(dataset):
                break

            indices = range(i - self.history_size, i, self.step)
            data.append(dataset[indices])

            if single_step:
                labels.append(dataset[i + self.target_size])
            else:
                labels.append(dataset[i:i + self.target_size])

        data_arr = np.array(data)
        labels_arr = np.array(labels)

        print(f'After reshaping, data: {data_arr.shape}, labels: {labels_arr.shape}')

        return data_arr, labels_arr

    def create_train_val_arr(
        self,
        arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
      
        x_train, y_train = self.multistep_data(
            dataset=arr,
            start_index=0,
            end_index=int(len(arr) * self.train_split),
        )
        x_val, y_val = self.multistep_data(
            dataset=arr,
            start_index=int(len(arr) * self.train_split),
            end_index=None,
        )

        return x_train, y_train, x_val, y_val

    def process_train_val_tf(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    
        train_multistep = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_multistep = train_multistep.cache().shuffle(self.buffer_size).batch(self.batch_size)  # .repeat()

        val_multistep = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_multistep = val_multistep.batch(self.batch_size)  # .repeat()

        return train_multistep, val_multistep

    def process(self, data) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        scaled_arr = self.scale_data(data)
        x_train, y_train, x_val, y_val = self.create_train_val_arr(scaled_arr)
        train_multistep, val_multistep = self.process_train_val_tf(x_train, y_train, x_val, y_val)
        return train_multistep, val_multistep
