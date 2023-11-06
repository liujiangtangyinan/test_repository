#############################
# 1. 首先重写metrics_pandas或metrics_pyspark的create_data_model和write_metrics方法
# 2. 更新config文件，定义指标计算的配置
# 3. 如在配置中使用了新的自定义函数，需要在user_define_funcs.py文件中加入该自定义函数的定义信息
# 4. 修改main函数的output, grouper等计算信息
# 5. 执行计算
#
##########################
from metrics_factory import MetricsFactory


def main():
    calc_mode = "pandas"
    # data model 函数需要自己实现，
    config_path = "metrics_config.csv"
    output = r"C:\work\tmp\featurelib_data\result6.csv"
    grouper = ["assets_wind_turbine_key"]
    period = []
    def create_data_model():
        dim_columns = ["assets_wind_turbine_key", "assets_wind_turbine_id", "blade_edge_moment_max_envelop",
                       "blade_edge_moment_min_envelop", "blade_flap_moment_max_envelop",
                       "blade_flap_moment_min_envelop"]
        cols = ["wind_farm_key", "calendar_key", "wind_farm_time", "design_wind_turbine_key", "ts_file"]
        import pandas as pd
        df_dt_load = pd.read_parquet(r"C:\work\tmp\featurelib_data\DigitalTwin\10MinLoad\event_date=2021-04-01")
        df_dt_load.drop(columns=cols, inplace=True)
        # 计算过程仅需assets_wind_turbine_id。 其实这里用key也行...
        df_dim_assets_wind_turbine = pd.read_parquet(r"C:\work\tmp\featurelib_data\AssetsWindTurbine")[dim_columns]
        cond = ["assets_wind_turbine_key"]
        pdf = pd.merge(df_dt_load, df_dim_assets_wind_turbine, on=cond)
        pdf['hour'] = pdf['plc_time'].dt.hour
        return pdf

    # def create_data_model():
    #     from pyspark.sql import SparkSession
    #     spark = SparkSession \
    #         .builder \
    #         .master("local") \
    #         .appName("calc frozen feature lib") \
    #         .config("spark.sql.broadcastTimeout", "1800") \
    #         .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    #         .getOrCreate()
    #
    #     dim_columns = ["assets_wind_turbine_key", "assets_wind_turbine_id", "blade_edge_moment_max_envelop",
    #                    "blade_edge_moment_min_envelop", "blade_flap_moment_max_envelop",
    #                    "blade_flap_moment_min_envelop"]
    #     cols = ["wind_farm_key", "calendar_key", "wind_farm_time", "design_wind_turbine_key", "ts_file"]
    #
    #     df_dt_load = spark.read.load(r"C:\work\tmp\featurelib_data\DigitalTwin\10MinLoad\event_date=2021-04-01",
    #                                  format="parquet").drop(*cols)
    #     df_dim_assets_wind_turbine = spark.read.load(
    #         r"C:\work\tmp\featurelib_data\AssetsWindTurbine\dim_assetswindturbine.parquet",
    #         format="parquet").select(dim_columns)
    #     cond = ["assets_wind_turbine_key"]
    #     return df_dt_load.join(df_dim_assets_wind_turbine, ["assets_wind_turbine_key"], "inner")

    data_model = create_data_model()
    metric_factory = MetricsFactory()
    metrics = metric_factory.create_metrics(config_path, None, grouper, period, data_model)
    metrics.metrics_apply()
    metrics.write_metrics(output, "csv")


if __name__ == "__main__":
    main()
