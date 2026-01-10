import json
import yaml
from kafka import KafkaProducer

CONFIG_PATH = "src/config/kafka.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

producer = KafkaProducer(
    bootstrap_servers=cfg["bootstrap_servers"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

TOPIC = cfg["topics"]["weather_input"]

def send_weather_event(payload: dict):
    """
    Payload must match PredictRequest schema
    """
    producer.send(TOPIC, payload)
    producer.flush()
