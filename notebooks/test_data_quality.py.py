# Databricks notebook source
import pytest
from pyspark.sql import SparkSession

# ------------- Pytest Fixtures -------------

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.getOrCreate()

# ------------- Bronze Layer Tests -------------

def test_bronze_transactions_nonempty(spark):
    df = spark.table("grocery_catalog.raw.transactions")
    assert df.count() > 0, "Bronze transactions table is empty."

def test_bronze_schema_columns(spark):
    df = spark.table("grocery_catalog.raw.transactions")
    required_columns = {"date", "store_nbr", "transactions"}
    assert required_columns.issubset(df.columns), \
        f"Bronze table missing columns: {required_columns - set(df.columns)}"

def test_bronze_no_nulls_in_key_columns(spark):
    df = spark.table("grocery_catalog.raw.transactions")
    nulls = df.filter(
        df["date"].isNull() | df["store_nbr"].isNull() | df["transactions"].isNull()
    ).count()
    assert nulls == 0, "Bronze table has nulls in key columns."

# ------------- Silver Layer Tests -------------

def test_silver_cleaned_nonempty(spark):
    df = spark.table("grocery_catalog.processed.sales_cleaned")
    assert df.count() > 0, "Silver cleaned table is empty."

def test_silver_no_nulls_any_column(spark):
    df = spark.table("grocery_catalog.processed.sales_cleaned")
    for col_name in df.columns:
        nulls = df.filter(df[col_name].isNull()).count()
        assert nulls == 0, f"Column '{col_name}' in Silver table has nulls."

def test_silver_schema_alignment(spark):
    df = spark.table("grocery_catalog.processed.sales_cleaned")
    expected_columns = {
        "date", "store_nbr", "transactions", "city", "state", "type", "cluster",
        "is_holiday", "dcoilwtico", "year", "month"
    }
    missing_cols = expected_columns - set(df.columns)
    assert not missing_cols, f"Missing columns in Silver table: {missing_cols}"

# ------------- Gold Layer Tests -------------

def test_gold_features_nonempty(spark):
    df = spark.table("grocery_catalog.analytics.sales_forecast_features")
    assert df.count() > 0, "Gold features table is empty."

def test_gold_feature_columns(spark):
    df = spark.table("grocery_catalog.analytics.sales_forecast_features")
    gold_expected = {
        "store_nbr", "year", "week", "weekly_transactions", "city", "state", "store_type",
        "store_cluster", "had_holiday", "avg_oil_price", "prev_week_transactions",
        "prev2_week_transactions", "mean_last4_weeks"
    }
    missing = gold_expected - set(df.columns)
    assert not missing, f"Missing features in Gold table: {missing}"

def test_gold_no_nulls_in_essential_features(spark):
    df = spark.table("grocery_catalog.analytics.sales_forecast_features")
    essential = ["store_nbr", "year", "week", "weekly_transactions"]
    for col_name in essential:
        nulls = df.filter(df[col_name].isNull()).count()
        assert nulls == 0, f"Essential feature '{col_name}' in Gold table has nulls."

def test_gold_weekly_transaction_range(spark):
    df = spark.table("grocery_catalog.analytics.sales_forecast_features")
    stats = df.agg({"weekly_transactions": "min"}).collect()[0][0]
    assert stats is not None and stats >= 0, "Negative or null weekly_transactions found in Gold."

# ------------- Integration & Range Tests -------------

def test_row_counts_increasing(spark):
    bronze = spark.table("grocery_catalog.raw.transactions").count()
    silver = spark.table("grocery_catalog.processed.sales_cleaned").count()
    gold = spark.table("grocery_catalog.analytics.sales_forecast_features").count()
    assert gold < silver <= bronze, \
        "Row counts should decrease from Bronze → Silver → Gold due to aggregation."

def test_sales_consistency(spark):
    bronze_total = spark.table("grocery_catalog.raw.transactions").agg({"transactions": "sum"}).collect()[0][0]
    gold_total = (
        spark.table("grocery_catalog.analytics.sales_forecast_features")
        .agg({"weekly_transactions": "sum"}).collect()[0][0]
    )
    assert gold_total <= bronze_total, "Gold aggregated transactions should not exceed Bronze total."