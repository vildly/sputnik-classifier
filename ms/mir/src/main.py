import os
import json
import nltk
from dotenv import load_dotenv
from kafka import KafkaProducer, producer

from clean import filter_data
from db import add_one, connect_to_database, find_one, get_collection


def create_producer() -> KafkaProducer:
    # Instantiate a KafkaProducer that connects to the Docker Kafka broker on 0.0.0.0:9002.
    # The value_serializer function converts a Python object to JSON-encoded bytes.
    producer = KafkaProducer(
        bootstrap_servers=["0.0.0.0:9002"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    return producer


def send_message(producer: KafkaProducer, topic, message):
    # Send a message to the specified topic and wait for confirmation.
    future = producer.send(topic, message)
    result = future.get(timeout=10)
    print(
        f"Message sent to topic {result.topic} in partition {result.partition} with offset {result.offset}"
    )


def get_dependencies() -> None:
    # Important as NLTK searches in these directories:
    # - $HOME/nltk_data
    # - $HOME/<path-to-project>/.venv/nltk_data
    # - $HOME/<path-to-project>/.venv/share/nltk_data
    # - $HOME/<path-to-project>/.venv/lib/nltk_data
    # - /usr/share/nltk_data
    # - /usr/local/share/nltk_data
    # - /usr/lib/nltk_data
    # - /usr/local/lib/nltk_data
    download_dir = "./.venv/lib/nltk_data"
    nltk.download("stopwords", download_dir=download_dir)
    nltk.download("punkt_tab", download_dir=download_dir)
    nltk.download("punkt", download_dir=download_dir)


if __name__ == "__main__":
    load_dotenv()
    connect_to_database(os.getenv("MONGODB_URI"))

    topic = "events"
    message = {"example": "Kafka message from python producer"}

    producer = create_producer()
    send_message(producer, topic, message)

    # Ensure all buffered messages are sent
    producer.flush()

    # col = get_collection("db", "preprocessed")
    # print(add_one(col=col, data={"data": "hello world"}))
    # print(find_one(col=col, query={"data": "hello world"}))
