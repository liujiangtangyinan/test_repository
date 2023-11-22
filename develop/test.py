#!/usr/bin/python3
# -*- coding:utf-8 -*-

# 导入必要的模块
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
from pyspark.sql.functions import pandas_udf, ceil

# 创建 SparkSession
spark = SparkSession.builder.appName("YourAppName").getOrCreate()
df = spark.createDataFrame([
    ("Alice", 2022, "Math", 90),
    ("Alice", 2022, "Science", 95),
    ("Bob", 2022, "Math", 85),
    ("Bob", 2023, "Math", 85),
    ("Bob", 2022, "Science", 92),
    ("Bob", 2023, "Science", 92),
    ("Charlie", 2022, "Math", 88),
    ("Charlie", 2022, "Science", 90)
], ["Name", "Year", "Subject", "Score"])

# df.show()
col_list = ["Name", "Year"]



def convert(df):
    def agg_udf(key, pdf):
        res = pdf.pivot(columns='Subject', values='Score')
        print(key, res)
        # for col in col_list:
        #     res[col] = pdf[col]
        res[col_list] = pdf[col_list]
        res['English'] = np.nan
        print(res.dtypes)
        return res
    pivoted_df = df.groupBy(*col_list).applyInPandas(agg_udf, "Year long, Name string, Math long, Science long, English long")
    # pivoted_df = df.groupBy(*col_list).agg(max/min/first(col))
    return pivoted_df

result_df = convert(df)
result_df.show(truncate=False)

print(1)