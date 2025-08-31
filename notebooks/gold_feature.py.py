from pyspark.sql import SparkSession
from pyspark.sql.functions import weekofyear, year, first, sum as spark_sum, max as spark_max, avg, lag, col
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType
import pandas as pd
import lightgbm as lgb

spark = SparkSession.builder.getOrCreate()
sales_cleaned = spark.table("grocery_catalog.processed.sales_cleaned")

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

window_spec = Window.partitionBy("store_nbr").orderBy("year", "week")
sales_weekly = (
    sales_weekly
    .withColumn("prev_week_transactions", lag("weekly_transactions", 1).over(window_spec))
    .withColumn("prev2_week_transactions", lag("weekly_transactions", 2).over(window_spec))
)


sales_weekly = sales_weekly.withColumn(
    "mean_last4_weeks",
    avg("weekly_transactions").over(window_spec.rowsBetween(-4, -1))
)

sales_weekly = sales_weekly.na.drop(subset=["prev_week_transactions", "prev2_week_transactions", "mean_last4_weeks"])


sales_weekly = (
    sales_weekly
    .withColumn("had_holiday", col("had_holiday").cast(IntegerType()))
    .withColumn("avg_oil_price", col("avg_oil_price").cast(DoubleType()))
    .withColumn("prev_week_transactions", col("prev_week_transactions").cast(DoubleType()))
    .withColumn("prev2_week_transactions", col("prev2_week_transactions").cast(DoubleType()))
    .withColumn("mean_last4_weeks", col("mean_last4_weeks").cast(DoubleType()))
    .withColumn("weekly_transactions", col("weekly_transactions").cast(DoubleType()))
)

feature_cols = ["prev_week_transactions", "prev2_week_transactions", "mean_last4_weeks", "had_holiday", "avg_oil_price"]
sales_pd = sales_weekly.select(feature_cols + ["weekly_transactions", "store_nbr", "year", "week", "city", "state", "store_type", "store_cluster"]).toPandas()

X = sales_pd[feature_cols]
y = sales_pd["weekly_transactions"]

lgb_train = lgb.Dataset(X, label=y)
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1
}
model = lgb.train(params, lgb_train, num_boost_round=100)

sales_pd["predicted_transactions"] = model.predict(X)

sales_predictions = spark.createDataFrame(sales_pd)

delta_output_path = "s3://grocery-sales-data-0067/grocery_sales/sales_forecast_features"

sales_predictions.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("year") \
    .save(delta_output_path)

print(f"Gold layer complete - features and ML predictions saved to: {delta_output_path}")

