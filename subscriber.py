from typing import Callable, Optional
from model_manager import ModelManager
from collections import deque
from config import Config
import numpy as np
import threading
import logging
import redis
import signal
import json
import time
import sys

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0  # Default Redis DB
CHANNEL = 'dht11:temperature'
REAL_TIME = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Subscriber:
    def __init__(
        self,
        channel: str,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        real_time: bool = REAL_TIME,
        prediction_callback: Optional[Callable[[deque, Optional[object]], None]] = None,
        model: Optional[object] = None,
        scaler: Optional[object] = None,
        config = Config('config.json'),
    ):
        self.channel = channel
        self.history_size = config.history_size
        self.real_time = real_time
        self.model = model  
        self.scaler = scaler
        self.prediction_callback = prediction_callback 
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(self.channel)

        self.history_buffer = deque(maxlen=self.history_size)
        self._stop_event = threading.Event()
        self.listener_thread = threading.Thread(target=self.listen_redis, daemon=True)

    def start(self):
        logger.info("Starting Subscriber.")
        self.listener_thread.start()
        self._setup_signal_handlers()

    def stop(self):
        logger.info("Stopping Subscriber.")
        self._stop_event.set()
        self.pubsub.close()
        self.listener_thread.join()
        logger.info("Subscriber stopped.")

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info(f"Received signal {sig}. Initiating shutdown.")
        self.stop()
        sys.exit(0)

    def listen_redis(self):
        logger.info(f"Subscribed to Redis channel: {self.channel}")
        for message in self.pubsub.listen():
            if self._stop_event.is_set():
                break

            if message['type'] == 'message':
                data = message['data']
                self.process_message(data)

    def process_message(self, data: str):
        """
        :param data: The message data as a string.
        """
        try:
            temperature = float(data)
            self.history_buffer.append(temperature)
            logger.info(f"Received temperature: {temperature}")

            if self.real_time and len(self.history_buffer) == self.history_size:
                logger.info("History buffer full. Initiating real-time prediction.")
                prediction_thread = threading.Thread(
                    target=self.prediction_callback,
                    args=(self.history_buffer.copy(), self.model, self.scaler, self.history_size),  # Pass history_size
                    daemon=True
                )
                prediction_thread.start()
        except ValueError:
            logger.warning(f"Invalid data received: {data}")



def make_real_time_prediction(data: deque, model: Optional[object], scaler: Optional[object], history_size):
    """
    :param data: A deque containing the history of temperature readings.
    :param model: The model for making predictions.
    """
    logger.info(f"Making real-time prediction with {len(data)} data points.")
    if model:
        logger.info("Using model to make predictions.")
        arr = np.array(data)
        scaled_arr = (arr - scaler['mean']) / scaler['std']
        reshaped_arr = scaled_arr.reshape((1, history_size, 1))       
        with open("predictions.txt", "a") as f:
            f.write(f"{model.predict(reshaped_arr)}\n")
    else:
        logger.warning("No model provided for predictions.")


def main():
    config = Config('config.json')  
    modelManager = ModelManager(config, 'LSTM')
    lstmModel = modelManager.build_model()
    lstmScaler = modelManager.load_scaler()

    subscriber = Subscriber(
        channel=CHANNEL,
        prediction_callback=make_real_time_prediction,
        model=lstmModel,
        scaler=lstmScaler
    )
    subscriber.start()

    try:
        while True:
            time.sleep(1) # Keep the main thread alive
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
        subscriber.stop()


if __name__ == "__main__":
    main()
