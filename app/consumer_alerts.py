import json
import os
from kafka import KafkaConsumer

BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    or os.getenv("KAFKA_BOOTSTRAP")
    or "kafka:9092"
)
TOPIC = "toxic_alerts"

def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )

    print(f"Listening on {TOPIC} ...")
    for msg in consumer:
        v = msg.value
        score = v.get("toxicity_score")
        print("\n[ALERT] Toxic detected")
        print("model:", v.get("model"))
        if score is not None:
            try:
                print("score:", round(float(score), 4))
            except (TypeError, ValueError):
                print("score:", score)
        print("text :", (v.get("comment_text") or "")[:200])

if __name__ == "__main__":
    main()
