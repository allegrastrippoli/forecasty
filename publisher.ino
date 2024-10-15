#include <WiFi.h>
#include <ArduinoJson.h>
#include <DHT11.h>

#define DHT11PIN 13  
DHT11 dht11(DHT11PIN);

const char* ssid = "XXX";
const char* password = "XXX";

const char* redisHost = "192.168.10.33"; 
const uint16_t redisPort = 6379;         
const char* redisPassword = NULL;        

WiFiClient redisClient;

float readTemperature() {
    int temperature = 0;
    int humidity = 0;

    int result = dht11.readTemperatureHumidity(temperature, humidity);

    if (isnan(temperature)) {
        Serial.println("Unable to read from DHT sensor");
    } else {
        Serial.print("Temperature = ");
        Serial.println(temperature);
    }
    return temperature;
}

bool connectToRedis() {
    if (!redisClient.connected()) {
        Serial.print("Connecting to Redis ");
        Serial.print(redisHost);
        Serial.print(":");
        Serial.println(redisPort);
        if (redisClient.connect(redisHost, redisPort)) {
            Serial.println("Connected to Redis");

            if (redisPassword != NULL && strlen(redisPassword) > 0) {
                String authCommand = String("*2\r\n$4\r\nAUTH\r\n$") + String(strlen(redisPassword)) + "\r\n" + String(redisPassword) + "\r\n";
                redisClient.print(authCommand);

                while (!redisClient.available()) {
                    delay(10);
                }
                String response = redisClient.readStringUntil('\n');
                response.trim();
                if (response != "+OK") {
                    Serial.println("Redis authentication failed");
                    redisClient.stop();
                    return false;
                }
                Serial.println("Redis authentication succeeded");
            }

            return true;
        } else {
            Serial.println("Failed to connect to Redis");
            return false;
        }
    }
    return true;
}

bool publishToRedis(const char* channel, const char* message) {
    if (!connectToRedis()) {
        return false;
    }

    String publishCommand = String("*3\r\n$7\r\nPUBLISH\r\n$") + String(strlen(channel)) + "\r\n" + String(channel) + "\r\n$" + String(strlen(message)) + "\r\n" + String(message) + "\r\n";

    redisClient.print(publishCommand);

    unsigned long timeout = millis();
    while (!redisClient.available()) {
        if (millis() - timeout > 2000) { 
            Serial.println("Timeout while publishing to Redis");
            redisClient.stop();
            return false;
        }
        delay(10);
    }

    String response = redisClient.readStringUntil('\n');
    response.trim();

    if (response.startsWith(":")) {
        Serial.print("Message published to Redis, subscribers: ");
        Serial.println(response.substring(1));
        return true;
    } else {
        Serial.print("Error while publishing to Redis: ");
        Serial.println(response);
        redisClient.stop();
        return false;
    }
}

void setup() {
    Serial.begin(115200);

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
}

void loop() {
    float temperature = readTemperature();
    char buf[10];
    snprintf(buf, 10, "%.1f", temperature);

    const char* channel = "dht11:temperature"; 
    if (publishToRedis(channel, buf)) {
        Serial.println("Data successfully published to Redis");
    } else {
        Serial.println("Error publishing data to Redis");
    }

    delay(1000);  
}

