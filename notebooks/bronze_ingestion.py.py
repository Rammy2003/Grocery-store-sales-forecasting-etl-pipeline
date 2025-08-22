# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import to_date, year, month, input_file_name, current_timestamp, col
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Set paths and log table name
VOLUME_PATH = "/Volumes/workspace/default/default_volume/grocery_store"
LOG_TABLE = "grocery_catalog.logs.etl_errors"

def load_to_bronze(csv_file, schema, delta_table, bad_records_dir, partition_by_date=False):
    try:
        df = (
            spark.read
            .option("header", True)
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .option("badRecordsPath", bad_records_dir)
            .schema(schema)
            .csv(csv_file)
            .withColumn("source_file", col("_metadata.file_path"))
        )
        if partition_by_date and "date" in df.columns:
            df = df.withColumn("date", to_date(col("date").cast(StringType()), "yyyy-MM-dd")) \
                   .withColumn("year", year(col("date"))) \
                   .withColumn("month", month(col("date")))
            df.write.format("delta").mode("overwrite").partitionBy("year", "month").saveAsTable(delta_table)
        else:
            df.write.format("delta").mode("overwrite").saveAsTable(delta_table)

        print(f"Ingested {csv_file} to {delta_table}")
    except Exception as e:
        error_schema = StructType([
            StructField("error_message", StringType(), True),
            StructField("error_time", TimestampType(), True),
            StructField("stage", StringType(), True),
            StructField("source_file", StringType(), True)
        ])
        error_data = [
            (str(e), None, "bronze_ingestion", csv_file)
        ]
        error_df = spark.createDataFrame(error_data, schema=error_schema) \
            .withColumn("error_time", current_timestamp())
        error_df.write.format("delta").mode("append").saveAsTable(LOG_TABLE)
        print(f"Failed on {csv_file}: {e}")
        raise



# SCHEMA DEFINITIONS

stores_schema = StructType([
    StructField("store_nbr", IntegerType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("type", StringType(), True),
    StructField("cluster", IntegerType(), True)
])

sample_submission_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("sales", DoubleType(), True)
])

oil_schema = StructType([
    StructField("date", StringType(), True),
    StructField("dcoilwtico", DoubleType(), True)
])

holidays_events_schema = StructType([
    StructField("date", StringType(), True),
    StructField("type", StringType(), True),
    StructField("locale", StringType(), True),
    StructField("locale_name", StringType(), True),
    StructField("description", StringType(), True),
    StructField("transferred", StringType(), True)
])

transactions_schema = StructType([
    StructField("date", StringType(), True),
    StructField("store_nbr", IntegerType(), True),
    StructField("transactions", IntegerType(), True)
])

test_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("date", StringType(), True),
    StructField("store_nbr", IntegerType(), True),
    StructField("family", StringType(), True),
    StructField("onpromotion", IntegerType(), True)
])

# INGEST EACH FILE

load_to_bronze(
    csv_file=f"{VOLUME_PATH}/stores.csv",
    schema=stores_schema,
    delta_table="grocery_catalog.raw.stores",
    bad_records_dir=f"{VOLUME_PATH}/bad_records/stores"
)

load_to_bronze(
    csv_file=f"{VOLUME_PATH}/sample_submission.csv",
    schema=sample_submission_schema,
    delta_table="grocery_catalog.raw.sample_submission",
    bad_records_dir=f"{VOLUME_PATH}/bad_records/sample_submission"
)

load_to_bronze(
    csv_file=f"{VOLUME_PATH}/oil.csv",
    schema=oil_schema,
    delta_table="grocery_catalog.raw.oil",
    bad_records_dir=f"{VOLUME_PATH}/bad_records/oil",
    partition_by_date=True
)

load_to_bronze(
    csv_file=f"{VOLUME_PATH}/holidays_events.csv",
    schema=holidays_events_schema,
    delta_table="grocery_catalog.raw.holidays_events",
    bad_records_dir=f"{VOLUME_PATH}/bad_records/holidays_events",
    partition_by_date=True
)

load_to_bronze(
    csv_file=f"{VOLUME_PATH}/transactions.csv",
    schema=transactions_schema,
    delta_table="grocery_catalog.raw.transactions",
    bad_records_dir=f"{VOLUME_PATH}/bad_records/transactions",
    partition_by_date=True
)

load_to_bronze(
    csv_file=f"{VOLUME_PATH}/test.csv",
    schema=test_schema,
    delta_table="grocery_catalog.raw.test",
    bad_records_dir=f"{VOLUME_PATH}/bad_records/test",
    partition_by_date=True
)
