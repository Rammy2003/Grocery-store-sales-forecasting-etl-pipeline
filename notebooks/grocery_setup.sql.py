# Databricks notebook source
# MAGIC %sql
# MAGIC Create catalog if not exists grocery_catalog;

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog grocery_catalog;
# MAGIC     
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS raw;
# MAGIC CREATE SCHEMA IF NOT EXISTS processed;
# MAGIC CREATE SCHEMA IF NOT EXISTS analytics;
# MAGIC CREATE SCHEMA IF NOT EXISTS logs;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists logs.etl_errors (
# MAGIC   error_message string,
# MAGIC   error_time timestamp,
# MAGIC   stage string,
# MAGIC   source_file string
# MAGIC );