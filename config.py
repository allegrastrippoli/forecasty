import json

class Config:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            configs = json.load(f)

        # General parameters
        self.cache_dir = configs['cache_dir']
        self.cache_file = configs['cache_file']
        self.raw_data = configs['raw_data']
        self.timestamp_col = configs['timestamp_col']
        self.value_col = configs['value_col']

        # Data processing parameters
        self.history_size = configs['history_size']
        self.target_size = configs['target_size']
        self.batch_size = configs['batch_size']
        self.train_split = configs['train_split_ratio']
        self.step = configs['step']
        self.buffer_size = configs.get('buffer_size', 1000)
        self.shuffle_seed = configs.get('shuffle_seed', 42)

        # Model parameters
        self.model_dir = configs.get('model_dir', 'model')
        self.tensorboard_log_dir = configs.get('tensorboard_log_dir', 'logs/tensorboard')

        # Training parameters
        self.patience = configs['patience']
        self.epochs = configs['epochs']
        self.max_trials = configs['max_trials']
        self.executions_per_trial = configs['executions_per_trial']

        # Hyperparameter tuning ranges
        self.num_lstm_layers_min = configs.get('num_lstm_layers_min', 1)
        self.num_lstm_layers_max = configs.get('num_lstm_layers_max', 3)
        self.dropout_min = configs.get('dropout_min', 0.1)
        self.dropout_max = configs.get('dropout_max', 0.5)
        self.dropout_step = configs.get('dropout_step', 0.1)
        self.learning_rate_min = configs.get('learning_rate_min', 1e-5)
        self.learning_rate_max = configs.get('learning_rate_max', 1e-3)

        # Optimizer
        self.optimizer = configs.get('optimizer', 'RMSprop')
