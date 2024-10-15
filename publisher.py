import redis
import json
import requests
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Publisher:
    def __init__(self, host='localhost', port=6379, db=0, channel='dht11:temperature'):
        try:
            self.redis_client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)
            self.channel = channel
            self.redis_client.ping()
            logging.info(f"Connected to Redis at {host}:{port}, DB: {db}")
        except redis.RedisError as e:
            logging.error(f"Failed to connect to Redis: {e}")
            raise

    def publish(self, message):
        """
        :param message: The message to publish
        """
        try:
            self.redis_client.publish(self.channel, message)
            logging.info(f"Published message to '{self.channel}': {message}")
        except redis.RedisError as e:
            logging.error(f"Error publishing message to Redis: {e}")

def get_temperature(url):
    """
    :param url: The URL to fetch the temperature data from
    :return: The temperature value or None if an error occurs
    """
    try:
        # response = requests.get(url, timeout=1)  
        response = requests.get(url)      
        response.raise_for_status()
        data = response.json()
        temperature = data.get("temperature")
        if temperature is not None:
            logging.debug(f"Fetched temperature: {temperature}")
            return temperature
        else:
            logging.warning("Temperature key not found in the response.")
            return None
    except requests.RequestException as e:
        logging.error(f"Error fetching temperature: {e}")
        return None
    except ValueError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return None

def main():
    temperature_url = "http://localhost:8000/temperature"
    configs = json.load(open('config.json', 'r'))
    # publish_interval = 1 

    try:
        publisher = Publisher()
    except Exception as e:
        logging.critical("Exiting due to Redis connection failure.")
        return

    i = 0
    # while True:
    while i < configs['history_size']: # generate exactly the number of data points required for the model to make a single multistep prediction 
        print(f"Publishing temperature data: {i}")
        temperature = get_temperature(temperature_url)
        if temperature is not None:
            publisher.publish(str(temperature))
        i += 1
        # time.sleep(publish_interval)

if __name__ == "__main__":
    main()
