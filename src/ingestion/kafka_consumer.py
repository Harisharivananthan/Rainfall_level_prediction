import json
import yaml
from kafka import KafkaConsumer, KafkaProducer

from src.streaming.realtime_predictor import predict_from_event

CONFIG_PATH = "src/config/kafka.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

consumer = KafkaConsumer(
    cfg["topics"]["weather_input"],
    bootstrap_servers=cfg["bootstrap_servers"],
    group_id=cfg["consumer_group"],
    auto_offset_reset="latest",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers=cfg["bootstrap_servers"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

OUTPUT_TOPIC = cfg["topics"]["predictions"]

def start_consumer():
    print("Kafka consumer started...")
    for msg in consumer:
        event = msg.value

        result = predict_from_event(event)

        producer.send(OUTPUT_TOPIC, result)
        producer.flush()
