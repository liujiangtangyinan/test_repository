from input_parser_tool import read_yaml_file
import pandas as pd
from metrics_app.user_define_funcs import exec_user_func
import numpy as np
from types import FunctionType


def load_params(input_file_path):
    job_input = read_yaml_file(input_file_path)
    params = []
    for pdic in job_input['CustomParameters']:
        for p_name, p_value in pdic.items():
            params.append((p_name, p_value['Value']))
    return params


def get_full_table_name(env: str, table_name: str) -> str:
    return "pd_galileo_" + env + "." + table_name


def map_agg_functions(agg_func):
    def percentile(n):
        def percentile_(x):
            x.dropna(inplace=True)
            if x.empty:
                x = pd.Series([np.nan])
            ret = np.percentile(x, n)
            return ret

        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    reference_dictionary = {"mean": "AVG", "amax": "MAX", "amin": "MIN", "sum": "sum", "max": "max", "min": "min",
                            "count": "count", "np.max": "MAX", "np.min": "MIN", "np.mean": "AVG"}
    if not agg_func:
        return None

    # 检测是否为多重聚合
    if isinstance(agg_func, list):
        if len(agg_func) > 1:
            return None
    elif isinstance(agg_func, str):
        if agg_func[0] == 'P':
            return percentile(int(agg_func[1:]))
        else:
            return reference_dictionary.get(agg_func, None)
    elif isinstance(agg_func, FunctionType):
        return reference_dictionary.get(agg_func.__name__, None)
    else:
        return None


def calc_agg_metrics(pdf_metrics_config, period=None, grouper=None,  executor_type='pandas'):
    def get_sub_df(pdf, cols):
        exists_cols = [_ for _ in cols if _ in pdf.columns]
        return pdf[exists_cols]

    def spark_udf_wrapper(key, pdf):
        pd_result = udf_wrapper(pdf)
        for k, g in zip(key, grouper):
            pd_result[g] = k
        return pd_result

    # pdf does not include groupby keys
    def udf_wrapper(pdf):
        pdf_result = pd.DataFrame()
        if period:
            pdf_result[period[0]] = np.sort(pdf[period[0]].unique())
            # num = len(pdf_result)
        # fields是单个指标计算所需的数据列，已转化为技术元数据，``飘号也处理掉了

        for ydata, metrics, filter_str, ydata_fields, filter_fields, agg_func, data_type, metric_type in zip(
                pdf_metrics_config["ydata"],
                pdf_metrics_config[
                    "metrics"],
                pdf_metrics_config[
                    "filter"],
                pdf_metrics_config[
                    "ydata_fields"],
                pdf_metrics_config[
                    "filter_fields"],
                pdf_metrics_config[
                    "agg_func"],
                pdf_metrics_config["type"],
                pdf_metrics_config["metric_type"]):
            # drop na可能导致group没有值
            if period:
                fields = list(set(ydata_fields + filter_fields + period))
            else:
                fields = list(set(ydata_fields + filter_fields))
            if metric_type != 'derived':
                sub_pdf = get_sub_df(pdf, fields)
                if filter_str:
                    sub_pdf = sub_pdf.query(filter_str)
                    if period != None:
                        sub_pdf = sub_pdf[ydata_fields+period]
                if agg_func and (sub_pdf.shape[0] != 0):
                    agg_name = map_agg_functions(agg_func)
                    # pandas 当period_field为空应该不会报错
                    if period:
                        data_applyed = sub_pdf.groupby(period).apply(agg_name)
                    else:
                        data_applyed = sub_pdf.apply(agg_name)
                    if isinstance(data_applyed, pd.Series):
                        sub_pdf = pd.DataFrame(data_applyed).T
                    else:
                        sub_pdf = data_applyed
            else:
                sub_pdf = get_sub_df(pdf_result, fields)
            if '#' in ydata:
                temp_data_ser = exec_user_func(sub_pdf, ydata)
            else:
                temp_data_ser = sub_pdf.eval(ydata)
            #temp_data_ser = temp_data_ser.reset_index(drop=True)
            pdf_result = pd.concat([pdf_result, pd.DataFrame({metrics: temp_data_ser})], axis=1)
        return pdf_result

    if executor_type == "pandas":
        return udf_wrapper
    else:
        return spark_udf_wrapper
