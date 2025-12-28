from pathlib import Path
import os

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

KAFKA_BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    or os.getenv("KAFKA_BOOTSTRAP")
    or "kafka:9092"
)

IN_TOPIC = "comments_raw"
SCORED_TOPIC = "comments_scored"
ALERTS_TOPIC = "toxic_alerts"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "toxicity_spark_pipeline").resolve()
CHECKPOINT_DIR = (BASE_DIR / "_chk").resolve()

schema = StructType([
    StructField("id", StringType(), True),
    StructField("comment_text", StringType(), True),
    StructField("produced_at", DoubleType(), True),
])

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train it first with train_spark_model.py"
        )

    spark = SparkSession.builder.appName("toxicity-stream").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    model = PipelineModel.load(str(MODEL_PATH))

    # Read stream from Kafka
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", IN_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), schema).alias("j"))
        .select("j.*")
        .withColumn("processed_at", current_timestamp())
        .dropna(subset=["comment_text"])
    )

    # Score with the trained Spark ML pipeline
    scored = model.transform(parsed)

    scored = (
        scored
        .withColumn("probability_arr", vector_to_array(col("probability")))
        .withColumn("toxicity_score", col("probability_arr").getItem(1))
        .withColumn("is_toxic", col("prediction") == 1.0)
        .withColumn("model", lit("lr_hashing_tf_idf_v1"))
        .drop("probability_arr")
    )

    # Write all scored results to comments_scored
    scored_out = scored.select(
        to_json(struct(
            col("id"),
            col("produced_at"),
            col("processed_at"),
            col("model"),
            col("toxicity_score"),
            col("is_toxic"),
            col("comment_text"),
        )).alias("value")
    )

    q1 = (
        scored_out.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", SCORED_TOPIC)
        .option("checkpointLocation", str(CHECKPOINT_DIR / "comments_scored"))
        .outputMode("append")
        .start()
    )

    # Write only toxic alerts to toxic_alerts
    alerts = scored.filter(col("is_toxic"))

    alerts_out = alerts.select(
        to_json(struct(
            col("id"),
            col("processed_at"),
            col("model"),
            col("toxicity_score"),
            col("is_toxic"),
            col("comment_text"),
        )).alias("value")
    )

    q2 = (
        alerts_out.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", ALERTS_TOPIC)
        .option("checkpointLocation", str(CHECKPOINT_DIR / "toxic_alerts"))
        .outputMode("append")
        .start()
    )

    q1.awaitTermination()
    q2.awaitTermination()

if __name__ == "__main__":
    main()
