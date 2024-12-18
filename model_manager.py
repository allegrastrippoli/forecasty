from lstm import LSTM
from transformer import Transformer
import tensorflow as tf
import os
import pickle

class ModelManager:
    def __init__(self, config, model_name='LSTM'):
        self.config = config 
        self.model_name = model_name
        self.model_path = os.path.join(config.model_dir, 'model.keras') 
        self.scaler_path = os.path.join(config.model_dir, 'scaler.pkl') 

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        print("Loaded model")
        return model

    def save_model(self, model):
        model.save(self.model_path)
        print("Saved model")

    def load_scaler(self):
        with open(self.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            print("Scaler loaded")
        return scaler

    def build_model(self, train_multistep=None, val_multistep=None):
        if os.path.exists(self.model_path):
            model = self.load_model()
            return model
        
        if self.model_name == 'LSTM':
            lstmClass = LSTM(self.config) 
            lstmModel = lstmClass.create(train_multistep, val_multistep)
            self.save_model(lstmModel)
            return lstmModel

        if self.model_name == 'Transformer':
            transformerClass = Transformer(self.config) 
            transformerModel = transformerClass.create(train_multistep, val_multistep)
            self.save_model(transformerModel)
            return transformerModel
