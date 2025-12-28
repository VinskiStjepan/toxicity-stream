import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    or os.getenv("KAFKA_BOOTSTRAP")
    or "kafka:9092"
)
TOPIC = "comments_raw"

BASE_DIR = Path(__file__).resolve().parent

def resolve_csv_path():
    candidates = []
    env_path = os.getenv("CSV_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append((BASE_DIR / ".." / "data" / "train.csv").resolve())
    candidates.append((BASE_DIR / "data" / "train.csv").resolve())

    for path in candidates:
        if path.exists():
            return path

    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Training data not found. Tried: {joined}")

# How fast to send
SLEEP_SECONDS = 0.02 

def main():
    csv_path = resolve_csv_path()

    df = pd.read_csv(csv_path, usecols=["comment_text"])
    df = df.dropna(subset=["comment_text"]).reset_index(drop=True)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        acks="all",
    )

    print(f"Loaded {len(df)} comments. Streaming to topic '{TOPIC}' on {BOOTSTRAP}...")

    for i, row in df.iterrows():
        payload = {
            "id": str(uuid.uuid4()),
            "comment_text": str(row["comment_text"])[:5000],  # avoid huge messages
            "produced_at": time.time(),
        }
        producer.send(TOPIC, payload)

        if i % 1000 == 0 and i > 0:
            producer.flush()
            print(f"Sent {i} messages...")

        time.sleep(SLEEP_SECONDS)

    producer.flush()
    print("Done streaming.")
    producer.close()

if __name__ == "__main__":
    main()
