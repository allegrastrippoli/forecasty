from keras_tuner import HyperModel, RandomSearch
import tensorflow as tf
import json

configs = json.load(open('config.json', 'r'))
HISTORY_SIZE = configs['history_size']
TARGET_SIZE = configs['target_size']
PATIENCE = configs['patience']
EPOCHS = configs['epochs']
MAX_TRIALS = configs['max_trials']
EXECUTIONS_PER_TRIAL = configs['executions_per_trial']
       
class LSTM(HyperModel):
    def __init__(self, train_shape=(HISTORY_SIZE,1), patience=PATIENCE, epochs=EPOCHS, max_trials=MAX_TRIALS, executions_per_trial=EXECUTIONS_PER_TRIAL):
        super(LSTM, self).__init__()
        self.train_shape = train_shape
        self.patience = patience
        self.epochs = epochs
        self.tuner = RandomSearch(
        self,
        objective='val_mae', 
        overwrite=True,
        max_trials=max_trials,       
        executions_per_trial=executions_per_trial,  
        directory='model',
        project_name='lstm_tuning'
    )
        
    def build(self, hp):
        inputs = tf.keras.Input(shape=self.train_shape)
        
        num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3)
        x = inputs
        
        for i in range(num_lstm_layers):
            units = hp.Int(f'lstm_units_{i+1}', min_value=16, max_value=128, step=16)
            x = tf.keras.layers.LSTM(units, 
                                     return_sequences=(i < num_lstm_layers - 1), 
                                     dropout=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1), 
                                     name=f"lstm_layer_{i+1}")(x)
        
        outputs = tf.keras.layers.Dense(TARGET_SIZE, name="dense_layer")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm")
        
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss='mae',
            metrics=['mae']
        )
        return model
        
    def create(self, train_multistep, val_multistep):
        tuner = self.tuner
        
        tuner.search_space_summary()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = self.patience, restore_best_weights=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard("model/content/tb_logs")
        
        tuner.search(train_multistep, validation_data=val_multistep, epochs=self.epochs, callbacks=[early_stopping, tensorboard_callback])

        best_model = tuner.get_best_models(num_models=1)[0]

        best_model.summary()
        return best_model
        


