


class MetricsFactory:
    def create_metrics(self, config_path, transform_map, grouper, period, data_model):
        import pandas as pd

        if self._isinstance_of_pandas_dataframe(data_model):
            from metrics_pandas import PandasEdaMetircs
            return PandasEdaMetircs(config_path, transform_map, grouper, period, data_model)
        elif self._isinstance_of_pyspark_dataframe(data_model):
            # from metrics_pyspark import PySparkEdaMetircs
            from metrics_app.metrics_pyspark import PySparkEdaMetircs
            return PySparkEdaMetircs(config_path, transform_map, grouper, period, data_model)
        else:
            raise ValueError(f"Invalid data mode: {data_model}")

    @staticmethod
    def _isinstance_of_pandas_dataframe(df):
        import pandas as pd
        return isinstance(df, pd.DataFrame)

    @staticmethod
    def _isinstance_of_pyspark_dataframe(df):
        import pyspark
        return isinstance(df, pyspark.sql.dataframe.DataFrame)