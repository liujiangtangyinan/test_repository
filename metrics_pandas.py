from metrics import EdaMetrics
import pandas as pd
from util import calc_agg_metrics


class PandasEdaMetircs(EdaMetrics):
    def __init__(self, config_path, transform_map, grouper, period, data_model):
        super().__init__(config_path, transform_map, grouper, period)
        self.data_model = data_model

    # todo 功能需要增强，支持parquet，并定义parquet的schema
    def write_metrics(self, path, format="parquet"):
        # 需要增强
        def write_data(data, output_path, output_format):
            # 根据不同的格式选择不同的写出方式
            write_method = getattr(pd.DataFrame, f"to_{output_format}")

            # 写出数据到指定的文件路径和格式
            write_method(data, output_path)
        write_data(self.data, path, format)

    def return_metric(self):
        return self.data



    def gen_schema(self, config, grouper, period, data_model):
        pdf = config
        return_schema = dict()
        for g in (grouper + period):
            return_schema[g] = data_model.dtypes[g].name
        for ytype, ylabel in zip(pdf["type"], pdf["metrics"]):
            return_schema[ylabel] = ytype
        return return_schema

    def metrics_apply(self):
        result = self.data_model.groupby(self.grouper).apply(calc_agg_metrics(self.metrics_config, self.period))
        result = result.reset_index().drop("level_1", axis=1)
        schema = self.gen_schema(self.metrics_config, self.grouper, self.period, self.data_model)
        self.data = result.astype(schema)
