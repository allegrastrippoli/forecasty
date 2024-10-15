import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

configs = json.load(open('config.json', 'r'))
STEP = configs['step']

def create_time_steps(length):
    return list(range(-length, 0))

def multistep_plot(i, history, true_future, prediction):

    plt.figure(figsize=(18, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo', label='True Future')
    
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')

    plt.legend(loc='upper left')

    os.makedirs('figures', exist_ok=True)

    plt.savefig(f'figures/multistep{i}.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    

    
