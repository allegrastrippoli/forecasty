from data_loader import DataLoader
from data_processor import DataProcessor
from fastapi import FastAPI
from config import Config
import uvicorn
import json

configs = json.load(open('config.json', 'r'))
HISTORY_SIZE = configs['history_size']
i = 0
app = FastAPI()
config = Config('config.json')  
dataLoader = DataLoader(config)
df = dataLoader.load_data() 
dataProcessor = DataProcessor(config)
x_train, y_train, x_val, y_val = dataProcessor.create_train_val_arr(df.values)
train_multistep, val_multistep = dataProcessor.process_train_val_tf(x_train, y_train, x_val, y_val)
xvalues = next(val_multistep.as_numpy_iterator())[0]

@app.get("/temperature")
async def get_temperature():
    global i
    temp = xvalues[0][i%HISTORY_SIZE][0]
    i += 1
    return {"temperature":temp}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    

        

    
        
    

    
    
