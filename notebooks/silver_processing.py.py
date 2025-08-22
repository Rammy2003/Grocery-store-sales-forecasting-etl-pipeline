# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, when, lit, coalesce, year, month

spark = SparkSession.builder.getOrCreate()

# Read from Bronze
transactions = spark.table("grocery_catalog.raw.transactions")
stores = spark.table("grocery_catalog.raw.stores")
holidays = spark.table("grocery_catalog.raw.holidays_events")
oil = spark.table("grocery_catalog.raw.oil")

# 1. Clean transactions data: deduplicate, convert date, drop nulls in any column
transactions_silver = (
    transactions
    .dropDuplicates(['date', 'store_nbr'])
    .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    .na.drop()  # Remove rows with any null in any column
)

# 2. Clean stores data: deduplicate, drop nulls
stores_silver = (
    stores
    .dropDuplicates(["store_nbr"])
    .na.drop()
)

# 3. Clean holidays_events: convert date, set holiday flag, filter, drop nulls
holidays_silver = (
    holidays
    .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    .withColumn("is_holiday", when(col("type") == "Holiday", lit(True)).otherwise(lit(False)))
    .filter(col("transferred") == "FALSE")
    .select("date", "is_holiday")
    .dropDuplicates(["date"])
    .na.drop()
)

# 4. Clean oil data: convert date, cast price, drop nulls
oil_silver = (
    oil
    .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    .withColumn("dcoilwtico", col("dcoilwtico").cast("double"))
    .dropDuplicates(['date'])
    .na.drop()
)

# 5. Join all cleaned datasets
sales_cleaned = (
    transactions_silver
    .join(stores_silver, on="store_nbr", how="left")
    .join(holidays_silver, on="date", how="left")
    .join(oil_silver, on="date", how="left")
    .withColumn("is_holiday", coalesce(col("is_holiday"), lit(False)))
)

# Remove existing 'year' and 'month' columns if any
for c in ['year', 'month']:
    if c in sales_cleaned.columns:
        sales_cleaned = sales_cleaned.drop(c)

# Drop the duplicate 'source_file' column if present
if 'source_file' in sales_cleaned.columns:
    sales_cleaned = sales_cleaned.drop('source_file')

# Add partition columns
sales_cleaned = (
    sales_cleaned
    .withColumn("year", year(col("date")))
    .withColumn("month", month(col("date")))
)

# Drop rows with any nulls again after join to ensure clean data
sales_cleaned = sales_cleaned.na.drop()

# Write cleaned and partitioned data to Silver table
sales_cleaned.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .saveAsTable("grocery_catalog.processed.sales_cleaned")

# Show result
display(sales_cleaned)
