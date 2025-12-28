FROM docker.io/library/spark:3.5.1

USER root

RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && pip3 install --no-cache-dir numpy pandas kafka-python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
