import re
import pandas as pd


class EdaMetrics:
    def __init__(self, config_path, transform_map, grouper=None, period=None):
        self.metrics_config = self._load_metrics_config(config_path, transform_map)
        self.data = pd.DataFrame()
        self.data_model = None
        self.grouper = grouper
        self.period = period

    def write_metrics(self):
        pass

    def gen_schema(self, config, grouper, data_model):
        pass

    def metrics_apply(self):
        pass

    def _load_metrics_config(self, path: str, transform_map=None) -> pd.DataFrame:
        if transform_map is None:
            transform_map = dict()
        pdf_conf = pd.read_csv(path)
        pdf_conf = pdf_conf.dropna(subset=['metrics'])
        pdf_conf['filter'] = pdf_conf['filter'].fillna('')
        pdf_conf['agg_func'] = pdf_conf['agg_func'].fillna('')
        for column in pdf_conf.columns:
            if pdf_conf[column].dtype.name == 'object':
                pdf_conf[column] = pdf_conf[column].fillna('')
                pdf_conf[column] = pdf_conf[column].apply(lambda i: i.strip() if isinstance(i, str) else i)
        pdf_conf['ydata'], pdf_conf['ydata_fields'] = self._transform_channels(pdf_conf['ydata'], transform_map)
        pdf_conf['filter'], pdf_conf['filter_fields'] = self._transform_channels(pdf_conf['filter'], transform_map)
        return pdf_conf

    def _transform_channels(self, exprs, transform_map):
        channel_extract_pattern = re.compile("`(.*?)`")
        lst_new_exprs = []
        lst_channels = []
        for expr in exprs:
            new_expr = expr
            channels = []
            for _ in re.findall(channel_extract_pattern, expr):
                if _ in transform_map:
                    new_expr = new_expr.replace(_, transform_map[_])
                    channels.append(transform_map[_])
                else:
                    channels.append(_)
                # 去掉飘号
                new_expr = new_expr.replace("`", "")
            lst_channels.append(channels)
            lst_new_exprs.append(new_expr)
        return lst_new_exprs, lst_channels


