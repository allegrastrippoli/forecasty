from data_loader import DataLoader
from data_processor import DataProcessor
from model_manager import ModelManager
from utils import multistep_plot

if __name__ == "__main__":
    dataLoader = DataLoader()
    df = dataLoader.load_data() 
    dataProcessor = DataProcessor()
    train_multistep, val_multistep = dataProcessor.process(df)
    modelManager = ModelManager('LSTM', 'model/model.keras')
    lstmModel = modelManager.build_model(train_multistep, val_multistep)
    for i,(x, y) in enumerate(val_multistep.take(10)):
        multistep_plot(i, x[0], y[0], lstmModel.predict(x)[0])
    
    
