# from metrics import EdaMetrics
from metrics_app.metrics import EdaMetrics
from metrics_app.util import calc_agg_metrics
from pyspark.sql.types import *

spark_type_map = {
    "bigint": "integer",
    "int": "integer",
    "double": "double",
    "string": "string",
    "float": "double",
    "date": "date"
}


class PySparkEdaMetircs(EdaMetrics):
    def __init__(self, config_path, transform_map, grouper, period, data_model):
        super().__init__(config_path, transform_map, grouper, period)
        self.data_model = data_model

    def gen_schema(self, config, grouper, data_model):
        pdf = config
        return_schema: StructType = StructType()
        for g in grouper:
            return_schema.add(g, spark_type_map[dict(data_model.dtypes)[g]], True)
        for ytype, ylabel in zip(pdf["type"], pdf["metrics"]):
            return_schema.add(ylabel, ytype, True)
        return return_schema

    # todo 功能需要增强
    def write_metrics(self, mode="append", format=None, path=None, partitionBy=None, saveAsTable=None, name=None,
                      rePartition=None, dict_options=dict()):
        if rePartition:
            self.data = self.data.repartition(rePartition)
        if saveAsTable:
            try:
                self.data.write.options(**dict_options).saveAsTable(name=name, mode=mode, format=format, path=path,
                                                                    partitionBy=None)
            except Exception as e:
                print("error:", str(e))
        else:
            try:
                self.data.write.options(**dict_options).save(mode=mode, format=format, path=path, partitionBy=None)
            except Exception as e:
                print("error:", str(e))

    def metrics_apply(self):
        # 选出用到的列名
        need_channels = list(
            set(list(self.metrics_config.explode("ydata_fields")["ydata_fields"].dropna().values) + list(
                self.metrics_config.explode("filter_fields")["filter_fields"].dropna().values)))
        need_channels = need_channels + self.grouper
        print(need_channels)
        print(type(need_channels))

        self.data_model = self.data_model.select(need_channels)

        self.data = self.data_model.groupby(self.grouper).applyInPandas(
            calc_agg_metrics(self.metrics_config, self.period, self.grouper, "pySpark"),
            self.gen_schema(self.metrics_config, self.grouper, self.data_model))
        #

    def return_metric(self):
        return self.data
