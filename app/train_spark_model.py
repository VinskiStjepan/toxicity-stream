import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

BASE_DIR = Path(__file__).resolve().parent
MODEL_OUT = (BASE_DIR / "models" / "toxicity_spark_pipeline").resolve()  # folder will be created

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

def main():
    csv_path = resolve_csv_path()

    spark = SparkSession.builder.appName("toxicity-train").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Read only needed columns
    df = (
        spark.read.option("header", True).option("escape", "\"").csv(str(csv_path))
        .select("comment_text", "target")
        .dropna(subset=["comment_text", "target"])
    )

    # Convert target to double and build binary label
    df = df.withColumn("target", col("target").cast("double"))
    df = df.withColumn("label", when(col("target") >= 0.5, 1.0).otherwise(0.0))

    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

    tokenizer = RegexTokenizer(
        inputCol="comment_text",
        outputCol="tokens",
        pattern="\\W+",
        minTokenLength=2
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )

    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="tf",
        numFeatures=1 << 18  # 262,144 features
    )

    idf = IDF(
        inputCol="tf",
        outputCol="features"
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.0,
        elasticNetParam=0.0
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, lr])

    print("Training model...")
    model = pipeline.fit(train_df)

    print("Evaluating...")
    preds = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(preds)
    print(f"AUC = {auc:.4f}")

    print(f"Saving model to: {MODEL_OUT}")
    model.write().overwrite().save(str(MODEL_OUT))

    spark.stop()

if __name__ == "__main__":
    main()
