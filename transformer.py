import tensorflow as tf
from keras_tuner import HyperModel, RandomSearch

class Transformer(HyperModel):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.train_shape = (config.history_size // config.step, 1) 
        self.patience = config.patience
        self.epochs = config.epochs
        self.max_trials = config.max_trials
        self.executions_per_trial = config.executions_per_trial

        self.tuner = RandomSearch(
            self,
            objective='val_mae',
            overwrite=True,
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory=config.model_dir,
            project_name='transformer_tuning'
        )

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.train_shape)

        x = self.positional_encoding_layer(inputs)

        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=3)
        for i in range(num_transformer_blocks):
            x = self.transformer_block(
                x,
                head_size=hp.Int(f'head_size_{i+1}', min_value=16, max_value=64, step=16),
                ff_dim=hp.Int(f'ff_dim_{i+1}', min_value=32, max_value=256, step=32),
                dropout=hp.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.5, step=0.1),
            )

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.config.target_size, activation='linear', name="dense_layer")(x)

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mae',
            metrics=['mae']
        )
        return model

    def transformer_block(self, x, head_size, ff_dim, dropout):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=head_size // 8,
            key_dim=head_size
        )(x, x)
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        attention_output = tf.keras.layers.Dense(x.shape[-1])(attention_output)  
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

        ff_output = tf.keras.layers.Dense(ff_dim, activation="relu")(attention_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
        ff_output = tf.keras.layers.Dense(x.shape[-1])(ff_output)
        ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
        return ff_output

    def positional_encoding_layer(self, inputs):
        positions = tf.range(start=0, limit=self.train_shape[0], delta=1)
        encoded_positions = tf.keras.layers.Embedding(
            input_dim=self.train_shape[0],
            output_dim=self.train_shape[1]
        )(positions)
        encoded_positions = tf.expand_dims(encoded_positions, axis=0)  
        return inputs + encoded_positions

    def create(self, train_multistep, val_multistep):
        self.tuner.search_space_summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.config.tensorboard_log_dir)

        self.tuner.search(
            train_multistep,
            validation_data=val_multistep,
            epochs=self.epochs,
            callbacks=[early_stopping, tensorboard_callback]
        )

        best_model = self.tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        return best_model
