# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import weekofyear, year, first, sum as spark_sum, max as spark_max, avg, lag, col
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# Read cleaned (Silver) sales data
sales_cleaned = spark.table("grocery_catalog.processed.sales_cleaned")

# 1. Aggregate weekly store-level sales
sales_weekly = (
    sales_cleaned
    .withColumn("week", weekofyear(col("date")))
    .withColumn("year", year(col("date")))
    .groupBy("store_nbr", "year", "week")
    .agg(
        spark_sum("transactions").alias("weekly_transactions"),
        first("city").alias("city"),
        first("state").alias("state"),
        first("type").alias("store_type"),
        first("cluster").alias("store_cluster"),
        spark_max("is_holiday").alias("had_holiday"),
        avg("dcoilwtico").alias("avg_oil_price")
    )
)

# 2. Add lag features: previous week's sales for the same store
window_spec = Window.partitionBy("store_nbr").orderBy("year", "week")

sales_weekly = (
    sales_weekly
    .withColumn("prev_week_transactions", lag("weekly_transactions", 1).over(window_spec))
    .withColumn("prev2_week_transactions", lag("weekly_transactions", 2).over(window_spec))
)

# 3. Optional: Add rolling mean of last 4 weeks
sales_weekly = sales_weekly.withColumn(
    "mean_last4_weeks",
    avg("weekly_transactions").over(window_spec.rowsBetween(-4, -1))
)

# 4. Write Gold features table
sales_weekly.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("year") \
    .saveAsTable("grocery_catalog.analytics.sales_forecast_features")

print("Gold layer complete - features engineered for forecasting.")
