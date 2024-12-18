from data_loader import DataLoader
from data_processor import DataProcessor
from model_manager import ModelManager
from utils import multistep_plot
from config import Config 

if __name__ == "__main__":
    config = Config('config.json')  
    dataLoader = DataLoader(config)
    df = dataLoader.load_data()
    dataProcessor = DataProcessor(config)
    train_multistep, val_multistep = dataProcessor.process(df)
    modelManager = ModelManager(config)
    model = modelManager.build_model(train_multistep, val_multistep)
    
