from lstm import LSTM
import tensorflow as tf
import os 
import pickle

class ModelManager:
    def __init__(self, model_name, model_path, scaler_path= 'model/scaler.pkl'):
        self.model_name = model_name
        self.model_path = model_path
        self.scaler_path = scaler_path
        
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
            print(f"Scaler loaded")
        
        return scaler

    def build_model(self, train_multistep=None, val_multistep=None):
        if self.model_name == 'LSTM':
            if os.path.exists(self.model_path):
                model = self.load_model()
                return model
            else:
                lstmClass = LSTM()
                lstmModel = lstmClass.create(train_multistep, val_multistep)
                self.save_model(lstmModel)
                return lstmModel
            


