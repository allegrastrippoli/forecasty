from keras_tuner import HyperModel, RandomSearch
import tensorflow as tf
import math

class LSTM(HyperModel):
    def __init__(self, config):
        super(LSTM, self).__init__()
        
        self.config = config
        self.train_shape = (math.ceil(config.history_size / config.step), 1)
        self.patience = config.patience
        self.epochs = config.epochs
        
        self.tuner = RandomSearch(
            self,
            objective='val_mae',
            overwrite=True,
            max_trials=config.max_trials,
            executions_per_trial=config.executions_per_trial,
            directory=config.model_dir,
            project_name='lstm_tuning'
        )
    
    def build(self, hp):
        inputs = tf.keras.Input(shape=self.train_shape)
        
        num_lstm_layers = hp.Int(
            'num_lstm_layers', 
            min_value=self.config.num_lstm_layers_min, 
            max_value=self.config.num_lstm_layers_max
        )
        x = inputs
        
        for i in range(num_lstm_layers):
            units = hp.Int(
                f'lstm_units_{i+1}', 
                min_value=16, 
                max_value=128, 
                step=16
            )
            x = tf.keras.layers.LSTM(
                units, 
                return_sequences=(i < num_lstm_layers - 1),
                dropout=hp.Float(
                    'dropout', 
                    min_value=self.config.dropout_min, 
                    max_value=self.config.dropout_max, 
                    step=self.config.dropout_step
                ), 
                name=f"lstm_layer_{i+1}"
            )(x)
        
        outputs = tf.keras.layers.Dense(self.config.target_size, name="dense_layer")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm")
        
        learning_rate = hp.Float(
            'learning_rate', 
            min_value=self.config.learning_rate_min, 
            max_value=self.config.learning_rate_max, 
            sampling='log'
        )

        optimizer = getattr(tf.keras.optimizers, self.config.optimizer)(
            learning_rate=learning_rate
        )
        model.compile(
            optimizer=optimizer,
            loss='mae',
            metrics=['mae']
        )
        return model
    
    def create(self, train_multistep, val_multistep):
        tuner = self.tuner
        
        tuner.search_space_summary()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
            restore_best_weights=True
        )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.config.tensorboard_log_dir
        )
        
        tuner.search(
            train_multistep, 
            validation_data=val_multistep, 
            epochs=self.epochs, 
            callbacks=[early_stopping, tensorboard_callback]
        )
        
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        return best_model
