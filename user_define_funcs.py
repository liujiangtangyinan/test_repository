# -*- coding: utf-8 -*-
"""
    # @Time : 2023/02/09
    Description:自定义函数引用的工具函数类
"""
import numpy as np
import pandas as pd
from datetime import datetime


def exec_user_func(df, ufe):
    """
   无此配置
   """
    ret = df.eval(ufe.replace("#", ""), engine='python')
    return pd.Series(ret)


def _df_extre_max(data: pd.Series):
    data.dropna(inplace=True)
    # result_data = data.iloc[:, 1:]
    if not data.empty:
        # return result_data.max().max()
        return data.max()
    else:
        return 0


def deal_extre(data: pd.DataFrame, method: str):
    try:
        num = int(method[1:])
    except ValueError:
        num = 99
    result_data = _df_extre(data, num)
    return result_data


def _df_extre(data: pd.DataFrame, num: int):
    p99_list = []
    try:
        [p99_list.extend((data[i].abs()).to_list()) for i in data.columns]
    except TypeError:
        return 0
    except AttributeError:
        return 0
    except ValueError:
        return 0
    result_data = pd.Series(p99_list)
    result_data.dropna(inplace=True)
    if not result_data.empty:
        return np.percentile(result_data.to_list(), num)
    else:
        return 0


def deal_extre_max(data: pd.DataFrame):
    # if 'wtg_alias' in data:
    #     result_data = data.groupby("wtg_alias")
    # else:
    #     result_data = data
    dataT = data.T
    val_data = []
    index_val = []
    try:
        # for i in result_data:
        for i in dataT.iterrows():
            max_data = _df_extre_max(i[1])
            index_val.append(i[0])
            val_data.append(max_data)
    except Exception as e:
        pass
    val = max(val_data)
    return pd.Series(val)


def _eql_mean(data: pd.Series, m: int, method: str):
    try:
        if method == "mean":
            result = np.power(np.mean(np.power(data, m)), 1 / m)
        elif method == "sum":
            result = np.power(np.sum(np.power(data, m)), 1 / m)
        else:
            result = np.nan
    except OverflowError:
        result = np.nan
    return result


def bias_percentile(data, p: int):
    data.dropna(inplace=True)
    try:
        channel_data = np.array(data)
    except KeyError:
        return 0
    except ValueError:
        return 0
    except Exception:
        return 0
    mean_data = channel_data.mean()
    if mean_data != 0 and not pd.isnull(mean_data):
        process_data = pd.Series(abs(channel_data - mean_data))
        process_data.dropna(inplace=True)
        return np.percentile(process_data.to_list(), p)
    else:
        return 0


def four_a(data: list):
    data = pd.Series(data)
    data.dropna(inplace=True)
    return np.power(np.sum(np.power(data, 4)), 0.25)


def four_b(first_del: list, dir_ang: list):
    # 机舱方位角由遍历不重复方位角改为固定linspace(0,180,61)方位角集合；
    # 机舱方位角统一转换为（0~180）区间：对机舱方位角小于180度数据改为角度+180变为正值, 角度值大于180的减去180
    first_del = pd.Series(first_del)
    first_del.dropna(inplace=True)
    dir_ang = pd.Series(dir_ang)
    dir_ang.dropna(inplace=True)
    new_dir_ang = dir_ang.apply(_convert_angels)
    fixed_dir = np.linspace(0, 180, 61)
    finally_list = [
        np.power(np.sum(np.power(first_del * np.cos((np.array(new_dir_ang) - i) * np.pi / 180), 4)), 0.25)
        for i in fixed_dir]
    if finally_list:
        return np.array(finally_list).max()
    else:
        return 0


# def _four_b(first_del, DirAng):
#     first_del = pd.Series(first_del)
#     first_del.dropna(inplace=True)
#     DirAng = pd.Series(DirAng)
#     DirAng.dropna(inplace=True)
#     uniqe_dir = list(set(DirAng))
#     finally_list = [np.power(np.mean(np.power(first_del * np.cos((np.array(DirAng) - i) * np.pi / 180), 4)), 0.25)
#                     for i
#                     in uniqe_dir]
#     if finally_list:
#         return np.array(finally_list).max()
#     else:
#         return 0


def _convert_angels(x):
    if x < 0:
        return x + 180
    else:
        return x - 180 if x > 180 else x


def gbx_shmy_pro(data: pd.Series, over: int):
    '''
    gbx_shmy_pro，#gbx_shmy_pro(#ldd_sum(`Freeze_GBX_SHMy_LDDT_Sum`),over=4000)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
         para:
            配置文件给的常量(数据类型：int)
    Returns:
        result:
            (数据类型：pd.Series)
    Examples:
         *data =
         para = 5000
         result = (0, 0.35445544554455444)
    '''
    dtype1 = type(data[0])
    data_list = eval(data[0])
    series = pd.Series(data_list)
    dict_agg = {4000: 74,
                5000: 80,
                6000: 87}
    sub_series = series[dict_agg[over]:]
    x = sub_series.sum()
    y = series.sum()
    result = x / y
    return pd.Series(result)


def max(*data: pd.Series):
    '''
    max，#max(`Freeze_TOW_TTFx_DEL4_Acc`, `Freeze_TOW_TTFx_DEL4_Count`)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
    Returns:
        series:
            (数据类型：pd.Series)
    Examples:
         *data = ((1,2,3),(2,3,4))
         series = (2,3,4)
    '''
    len_data = len(data)
    if len_data == 1:
        return pd.Series(data[0].max())
    else:
        series = pd.DataFrame(data).T.max(axis=1)
    return series


def gbx_load_abs(*data: pd.Series, method="Pxx"):
    '''
    gbx_load_abs，#gbx_load_abs(`STE_BRoot_DeltaMyMax`,`STE_BRoot_DeltaMyMin`,method="max")
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        method:
            如果没有值则默认”Pxx“(数据类型：str)
    Returns:
        finally_data:
            (数据类型：pd.Series)
    Examples:
        用例1：
        method="max"
        *data = ((213,None,222), (1,-1,None))
        series = (222.0，1.0)
        用例2：
        method="P99"
        *data = ((213,123,222), (1,-1,12))
        series = 221.55
    '''
    temp_buffer = abs(pd.DataFrame(data).T)
    if method.startswith("P"):
        # temp_buffer.drop([i for i in temp_buffer.columns if "Delta" in i or "10min" in i], axis=1, inplace=True)
        finally_data = deal_extre(temp_buffer, method)
        return finally_data
    else:
        # print(f"data shape:{temp_buffer.shape}, cols:{temp_buffer.columns}, index:{temp_buffer.index}")
        finally_data = deal_extre_max(temp_buffer)
        return finally_data


def bld_load_over_count_max(*data: pd.Series, ble_load_threshold: pd.Series):
    '''
    bld_load_over_count_max，#bld_load_over_count_max(`Stat_BR1EdgeBM_Max`,`Stat_BR2EdgeBM_Max`,`Stat_BR3EdgeBM_Max`, ble_load_threshold=`blade_edge_moment_max_envelop`)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        ble_load_threshold:
            (数据类型：pd.Series)
    Returns:
        return_data.sum():
            (数据类型：int)
    Examples:
        用例1：
        ble_load_threshold = (4,3,4)
        *data = ((-1,-2,-3), (2,3,4),(3,,5,6))
        return_data.sum() = 2
    '''
    ble_load_df = pd.DataFrame(data).T
    sum_over_count = []
    for one_variable_name in ble_load_df.columns:
        bld_load_over_count = ble_load_df[one_variable_name] > ble_load_threshold
        sum_over_count.append(bld_load_over_count)
    return_data = pd.concat(sum_over_count, axis=1).sum(axis=1)
    return_data[return_data > 0] = 1
    return return_data.sum()


def bld_load_over_count_min(*data: pd.Series, ble_load_threshold: pd.Series):
    '''
    bld_load_over_count_min，#bld_load_over_count_min(`Stat_BR1EdgeBM_Min`,`Stat_BR2EdgeBM_Min`,`Stat_BR3EdgeBM_Min`,ble_load_threshold=`blade_edge_moment_min_envelop`)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        ble_load_threshold:
            (数据类型：pd.Series)
    Returns:
        return_data.sum():
            (数据类型：int)
    Examples:
        用例1：
        ble_load_threshold = (4,2,3)staticmethod
        *data = ((5,2,1), (1,2,3),(4,5,6))
        return_data.sum() = 2
    '''
    ble_load_df = pd.DataFrame(data).T
    sum_over_count = []
    for one_variable_name in ble_load_df.columns:
        bld_load_over_count = ble_load_df[one_variable_name] < ble_load_threshold
        sum_over_count.append(bld_load_over_count)
    return_data = pd.concat(sum_over_count, axis=1).sum(axis=1)
    return_data[return_data > 0] = 1
    return return_data.sum()


def bld_extreme_load(*data: pd.Series, agr_type: str):  # 待刘璇验证
    '''
    bld_extreme_load，#bld_extreme_load(`Stat_BR1EdgeBM_Max`,`Stat_BR2EdgeBM_Max`,`Stat_BR3EdgeBM_Max`,agr_type="max")
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        agr_type:
            (数据类型：str)
    Returns:
        ble_load_df.agg(type_name, axis=1):
            (数据类型：pd.Series)
    Examples:
        用例1：
        agr_type = max
        *data = ((213,12,15),(7,30,44),(2,3,4))
        ble_load_df.agg(type_name, axis=1) = (213,30,44)
    '''
    type_dict = {'max': np.max, 'min': np.min}
    ble_load_df = pd.DataFrame(data).T
    type_name = type_dict[agr_type]
    return ble_load_df.agg(type_name, axis=1)


def ldd_sum(*data: pd.Series):
    '''
    ldd_sum，#ldd_sum(`MBR_HertzStress_LDD_10min`)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
    Returns:
        pd.Series(result):
            (数据类型：pd.Series)
    Examples:
        用例1：
        *data = (str)((1,2,3),(1,4,5),(5,6,7))
        pd.Series(result) = '[7,12,15]'
    '''

    def ldd_sum_eval(x):
        try:
            res = eval(x)
        except Exception as e:
            res = 0
        return res

    pd_data = pd.DataFrame(data).T
    # pd like this
    #     GBX_SHMx_LDD
    # 0   [1, 2, 3]
    # 1   [4, 5, 6]
    pd_data.fillna(0, inplace=True)
    series_ldd = pd_data.apply(lambda x: ldd_sum_eval(x[0]), axis=1)
    if series_ldd.shape[0] > 0:
        np_sum = np.array([0])
        for i in series_ldd:
            # series_ldd 内可能有0长度的内容。
            try:
                np_sum = np_sum + np.array(i)
            except ValueError as ve:
                continue
        tmp_result = np_sum
    else:
        tmp_result = np.array([0])
    tmp_result = ",".join(str(x) for x in tmp_result)
    result = '[' + tmp_result + ']'
    return pd.Series(result)


def twr_viv_over(*data: pd.Series):
    '''
    twr_viv_over，#twr_viv_over(`TOW_VIVTimeX1`, `TOW_VIVTimeY1`)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
    Returns:
        pd.concat([data_x, data_y], axis=1).agg("sum", axis=1).sum():
            (数据类型：pd.Series)
    Examples:
        用例1：
        *data = ((100,20,80),(29,30,50))
        pd.concat([data_x, data_y], axis=1).agg("sum", axis=1).sum() = 3
    '''
    pd_xy = pd.DataFrame(data).T
    data_x = pd_xy[pd_xy.columns[0]] > 30
    data_y = pd_xy[pd_xy.columns[1]] > 30
    return pd.concat([data_x, data_y], axis=1).agg("sum", axis=1).sum()


def twr_bias_prctile(data: pd.Series, p=None):
    '''
    twr_bias_prctile，#twr_bias_prctile(`TOW_TowerFrequency`,p=100)
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        p:
            默认None(数据类型：int)
    Returns:
        finally_data:
            (数据类型：pd.Series)
    Examples:
        用例1：
        p = 100
        *data = (1, 2, 4, 5, 6)
        finally_data = TOW_TowerFrequency    2.52
    '''
    """percentile(abs(`channel`-mean(`channel`)),90)"""
    # input_data = pd.DataFrame(data).T
    # finally_data = input_data.apply(lambda x: bias_percentile(x, p))
    finally_data = bias_percentile(data, p)
    return pd.Series(finally_data)


def twr_fatigue_analysis(*data: pd.Series, out_type=None):  # 待解决
    '''
    twr_fatigue_analysis，#twr_fatigue_analysis(`TOW_TTFx_DEL4`,`Stat_NacellePosAve`,'noDir')
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        out_type:
            默认None(数据类型：int)
    Returns:
        finally_data:
            (数据类型：pd.Series)
    Examples:
        用例1：
        out_type = noDir
        *data = ((1,2,3),(2,4,6))
        result = 3.1463462836457885
        用例2：
        out_type = wdir
        *data = ((10,10,10),(5.5,5,6.6))
        result = 13.159664453985638
    '''
    # input_data = pd.DataFrame(data).T
    input_data = pd.DataFrame({"tow_del": data[0], "dir_ang": data[1]})
    return __tow_df(input_data, out_type)


def __tow_df(data: pd.DataFrame, level: str):
    return _choice_level(level, data)


"""
内嵌函数（不应该放在这里的）
"""


def _choice_level(out_type: str, data_full: pd.DataFrame):
    data_full.dropna(inplace=True)
    first_del, dirang = data_full['tow_del'].to_list(), data_full['dir_ang'].to_list()
    del data_full
    if out_type == "noDir":
        return four_a(first_del)
    elif out_type == 'wDir':
        return four_b(first_del, dirang)
    else:
        low_data = four_a(first_del)
        top_data = four_b(first_del, dirang)
        if low_data == 0 or np.isnan(low_data):
            return 0
        return round(top_data / low_data, 3)


def ste_eql(data: pd.Series, m=10, method='mean'):
    '''
    ste_eql，#ste_eql(`STE_MS_Sig_m10_DEL_HS1_0`,m=10,method="sum")
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        method:
            默认None(数据类型：int)
        m:
            默认10(数据类型：int)
    Returns:
        result_data:
            (数据类型：pd.Series)
    Examples:
        用例1：
        m = 10
        method = sum
        *data = (1,2,3)
        result = 3.005167303445029
    '''
    need_deal_data = data
    # print(f"need_deal_data:\n{need_deal_data}")
    # result_data = need_deal_data.apply(_eql_mean, m, method)
    result_data = _eql_mean(need_deal_data, m, method)
    # print(f"result_data:{result_data}")
    return result_data


def exec_user_func(df, ufe):
    """
   无此配置
   """
    ret = df.eval(ufe.replace("#", "@"), engine='python')
    return pd.Series(ret)


def extrapolate(*data: pd.Series, m=1):
    '''
    extrapolate，#extrapolate(#eql_sum(`Freeze_TOW_TBMy_DEL4_Acc`,m=4),#sum(`Freeze_TOW_TBMy_DEL4_Count`),method='ToDesignedLife',m=4),#extrapolate(#sum(`Freeze_PTH_Pitch1Travel_SUM`),#sum(`Freeze_PTH_Pitch1Travel_Count`),method='ToPassTime240')
    Args:
        *data:
            经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
        method:
            默认None(数据类型：str)配置里有赋值字符串
        m:
            默认1(数据类型：int)
    Returns:
        ret:
            (数据类型：pd.Series)
    Examples:
        用例1：
        m = 1
        method = None
        *data = (2022, 2, 3),(2022, 1, 3),(2022, 12, 3)
        result = (106276320.0, 1261440.0, 157680.0)
    '''
    ret = (data[0] ** m * data[2] * 365 * 144 / data[1]) ** (1 / m)
    return ret


def sum(*data: pd.Series):
    '''
     sum，#sum(`Freeze_PTH_Pitch3Travel_Sum`)
     Args:
         *data:
             经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
     Returns:
         result:
             (数据类型：pd.Series)
     Examples:
         用例1：
         *data = ((1, 2, 3),(1, 2, 3))
         result = (2,4,6)
     '''
    len_data = len(data)
    if len_data == 1:
        return pd.Series(data[0].sum(), name=data[0].name)
    else:
        return pd.DataFrame(data).sum(axis=1)


def eql_sum(data: pd.Series, m=4):
    '''
     eql_sum，#eql_sum (`STE_MS_Sig_m10_DEL_HS1_0`,m=10)
     Args:
         data:
             经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
         m:
             默认1(数据类型：int)
     Returns:
         result:
             (数据类型：float)
     Examples:
         用例1：
         m = 10
         data = (1,2,3)
         result = 3.005167303445029
     '''
    return (data ** m).sum() ** (1.0 / m)


def eql(*data: pd.Series, m: int):
    '''
      eql，#eql(#eql_sum(`Freeze_TOW_TTFx_DEL4_Acc`,m=4),#sum(`Freeze_TOW_TTFx_DEL4_Count`),m=4)
      Args:
          *data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
          m:
              默认1(数据类型：int)
      Returns:
          result:
              (数据类型：float)
      Examples:
          用例1：
          m = 10
          *data = ((4,5,6),(2,4,6))
          result = (3.363586, 3.535534, 3.833659)
      '''
    # TODO 先占位，逻辑需等晓欣确定后再补充
    return ((data[0] ** m) / data[1]) ** (1.0 / m)


def min(*data: pd.Series):
    '''
      min，#min(`Freeze_TOW_TTFx_DEL4_Acc`, `Freeze_TOW_TTFx_DEL4_Count`)
      Args:
          *data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：pd.Series)
      Examples:
          用例1：
          *data = ((4,5),(4,6))
          result = (4,5)
      '''
    len_data = len(data)
    if len_data == 1:
        data[0].dropna(inplace=True)
        return pd.Series(data[0].min())
    else:
        series = pd.DataFrame(data).T.min(axis=1)
    # print(f"min func input:{data}\noutput:{series}")
    return series


def mbr_stress_process(data: pd.Series):
    '''
      mbr_stress_process，#mbr_stress_process(`Freeze_MB_HertzStrss_LDD_Sum`)
      Args:
          *data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：float)
      Examples:
          用例1：
          *data =(str)((1, 2, 3, 4, 5, 6,7, 8,9),(1, 2, 3, 4, 5, 6,7, 8,9))
          result = 1787.5
      '''
    data_bin_count = eval(data[0])
    stress_bin = [1300, 1400, 1500, 1550, 1600, 1650, 1700, 1750, 1800]
    data_all = pd.Series(data_bin_count).sum()
    # data_all = data_bin_count.sum
    if data_all <= 0:
        return 1000
    # percentage_bin = np.array((data_bin_count / data_all if data_all > 0.001 else 0.001).values[0])
    percentage_bin = data_bin_count / data_all if data_all > 0.001 else 0.001
    val_max = 0
    val_max_before = 0
    id_min = 0
    id_max = 0
    for index, val in enumerate(percentage_bin):
        val_max += val
        if val_max < 0.95:
            id_min = index
            val_max_before = val_max
            continue
        elif val_max > 0.95:
            id_max = index
            break
        else:
            return stress_bin[index]
    val_min, val_maxs = stress_bin[id_min], stress_bin[id_max]
    percent_num = round((0.95 - val_max_before) / (val_max - val_max_before), 2)
    return val_min + (val_maxs - val_min) * percent_num


def mbr_overstress_process(data: pd.Series):
    '''
      mbr_overstress_process，#mbr_overstress_process(`Freeze_MB_HertzStrss_LDD_Sum`)
      Args:
          *data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：float)
      Examples:
          用例1：
          *data =(str)((1, 2, 3, 4, 5, 6,7, 8,9),(1, 2, 3, 4, 5, 6,7, 8,9))
          result = 64.28571428571428
      '''
    data_bin_count = eval(data[0])
    data_all = pd.Series(data_bin_count).sum()
    # data_all = data_bin_count.sum
    if data_all <= 0:
        return 0
    percentage_bin = data_bin_count / data_all if data_all > 0.001 else 0.001
    # percentage_bin = np.array((data_bin_count / data_all if data_all > 0.001 else 0.001).values[0])
    stress_value = percentage_bin[2:].sum() * 100
    return stress_value


def mbr_ratio_process(data: pd.Series):
    '''
      mbr_ratio_process，#mbr_ratio_process(#ldd_sum(`Freeze_MB_Ratio_LDD_Sum`))
      Args:
          *data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：float)
      Examples:
          用例1：
          *data =(str)((1, 2, 3,4),(2, 1, 4,4))
          result = 90.0
      '''
    data_bin_count = eval(data[0])
    data_all = pd.Series(data_bin_count).sum()
    # data_all = data_bin_count.sum()
    if data_all <= 0:
        return 0
    percentage_bin = data_bin_count / data_all if data_all > 0.001 else 0.001
    # percentage_bin = np.array((data_bin_count / data_all if data_all > 0.001 else 0.001).values[0])
    return (1 - percentage_bin[0]) * 100


def idmax(*data: pd.Series):
    '''
      idmax，#idmax(*data)
      Args:
          *data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：pd.Series)
      Examples:
          用例1：
          *data =((1,2,3),(2,3,4))  index=Freeze_BLE_Mbf1_Acc_DEL_m4, Freeze_BLE_Mbf1_Acc_DEL_m3
          result = (Freeze_BLE_Mbf1_Acc_DEL_m3， Freeze_BLE_Mbf1_Acc_DEL_m3， Freeze_BLE_Mbf1_Acc_DEL_m3)
      '''
    return pd.DataFrame(data).T.idxmax(axis=1)


def gbx_ldd_prctile(data: pd.Series, P: float, abs_flag=False):  # 其中num有配置值是99.9
    '''
    gbx_ldd_prctile，# gbx_ldd_prctile(#ldd_sum(`Freeze_GBX_SHMy_LDDT_Sum`),P=90)

    Args:
      *data:
          经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      num:
         (数据类型：float)

    Returns:
        result:
            (数据类型：float)
    Examples:
        用例1：
        num = 90
        data = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]'  index=Freeze_BLE_Mbf1_Acc_DEL_m4, Freeze_BLE_Mbf1_Acc_DEL_m3
        result = None
    '''
    # P 是百分比
    P = P / 100.0
    try:
        data_sum = eval(data[0])
    except Exception as e:
        return np.nan
    if len(data_sum) != 100:
        return np.nan
    ldd_bin = np.linspace(-6850, 8000, 100)
    if abs_flag:
        ldd_bin = list(map(abs, ldd_bin))
        # data_sum = abs(data_sum)
    ldd_data = pd.DataFrame({"ldd_bin": ldd_bin, "data_sum": data_sum}, dtype=np.float)
    ldd_data.sort_values("ldd_bin", ascending=True, inplace=True)
    try:
        ldd_sum_data = float(pd.Series(data_sum).sum())
    except TypeError as e:
        return np.nan
    if ldd_sum_data == 0:
        return np.nan
    for i in range(100):
        if i == 0:
            a = ldd_data.iloc[0]["data_sum"]
        else:
            a = ldd_data.iloc[:i + 1]["data_sum"].sum()
        per_a = a / ldd_sum_data
        if per_a == P:
            return ldd_data.iloc[i]["ldd_bin"]
        elif per_a > P:
            before_per_a = ldd_data.iloc[:i]["data_sum"].sum() / ldd_sum_data
            x_array = np.array([before_per_a, per_a])
            y_array = np.array([ldd_data.iloc[i - 1]["ldd_bin"], ldd_data.iloc[i]["ldd_bin"]])
            interp = np.interp(P, x_array, y_array)
            return interp


def gbx_eql_torque(data: pd.Series, count_data: pd.Series, torque_val: pd.Series, p=None, p_max=None):
    '''
    gbx_eql_torque，#gbx_eql_torque(#ldd_sum(`Freeze_GBX_SHMx_LDD_Sum`),#sum(`Freeze_GBX_SHMx_LDD_Count`),RatedTorque=RatedTorque,p='6.6',nl='5E+7')
        Args:
          data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
          count_data:
             (数据类型：pd.Series)
          torque_val:
             (数据类型：float)
          p:
             默认None(数据类型：str)
          p_max:
             默认None(数据类型：str)
        Returns:
            result:
                (数据类型：pd.Series)
        Examples:
            用例1：
            data =(str)(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100)  index=Freeze_BLE_Mbf1_Acc_DEL_m4, Freeze_BLE_Mbf1_Acc_DEL_m3
            count_data = 100
            torque_val = 2000
            p = 6.6
            p_max = 5E+7
            result = 3039.2178706497893
    '''
    try:
        torque_val = float(torque_val[0])
        data_sum = eval(data[0])
    except Exception as e:
        return pd.Series([np.nan])
    if count_data.values == 0 or len(data_sum) != 100:
        return pd.Series([np.nan])
    ldd_bin = np.linspace(torque_val * -1, torque_val * 2, 100)
    ldd_data = pd.DataFrame({"ldd_bin": ldd_bin, "data_sum": data_sum}, dtype=np.float)
    ldd_data.sort_values('ldd_bin', ascending=False, inplace=True)
    ldd_data["data_sum"] = ldd_data["data_sum"].map(
        lambda x: (x / (count_data.values / 144)) * 20 * 365 if count_data.values != 0 else 0)
    n1, n2 = 0, 0
    for i in range(100):
        if ldd_data["ldd_bin"].iloc[i] <= 0:
            return pd.Series([np.nan])
        if i == 0:
            n1 = ldd_data["data_sum"].iloc[0]
            n2 = n1 * np.power((ldd_data["ldd_bin"].iloc[0] / ldd_data["ldd_bin"].iloc[1]),
                               float(p)) + ldd_data["data_sum"].iloc[1]
            continue
        else:
            n1 = n2
            n2 = n1 * np.power((ldd_data["ldd_bin"].iloc[i] / ldd_data["ldd_bin"].iloc[i + 1]),
                               float(p)) + ldd_data["data_sum"].iloc[i + 1]
        if n2 > float(p_max):
            return ldd_data["ldd_bin"].iloc[i] + (ldd_data["ldd_bin"].iloc[i + 1] - ldd_data["ldd_bin"].iloc[i]) * (
                    float(p_max) - n1) / (n2 - n1)


def div(data: pd.Series, dividend: float):
    '''
      idmax，#idmax(*data)
      Args:
          data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：pd.Series)
      Examples:
          用例1：
          data =((1,2,3))
         dividend = 2
          result = 0.5,1,1.5

      '''
    return data / dividend


def back_dim_keyword(data: pd.Series):
    '''
      此函数不改变值，为了增加输出通道为后期添加维度信息做准备。
      back_dim_keyword，#back_dim_keyword(`assets_wind_turbine_key`)
      Args:
          data:
              经过过滤和聚合后的风机资产数据(数据类型：pd.Series)
      Returns:
          result:
              (数据类型：pd.Series)
      Examples:
          用例1：
          data =[CN-53/23-B-016]
          result =[CN-53/23-B-016]
    '''
    return data


#
# def timestamp_sub(*data: pd.Series):
#     timestamp1_series = data[0]
#     timestamp2_series = data[1]
#     timestamp1 = timestamp1_series.iloc[0]
#     timestamp2 = timestamp2_series.iloc[0]
#     if timestamp2 ==None or timestamp1 == None:
#         return np.nan
#     if timestamp2 < timestamp1:
#         timestamp2, timestamp1 = timestamp1, timestamp2
#     # 时间戳相减，然后算出天数
#     day = (timestamp2 - timestamp1).days
#     return day


def timestamp_sub(date1: pd.Series, date2: pd.Series):
    try:
        day = ((date1 - date2) / (24 * 3600)).abs().astype(int)
    except pd.errors.IntCastingNaNError as ve:
        day = pd.Series([np.nan])
    return day


def windspeed_sub(*data: pd.Series, method=None):
    WindSpeed1 = data[0]
    WindSpeed2 = data[1]
    temp_buffer = WindSpeed1 - WindSpeed2
    # try:
    #     torque_val = float(torque_val[0])
    #     data_sum = eval(data[0])
    # except Exception as e:
    #     return pd.Series([np.nan])
    if method.startswith("P"):
        num = int(method[1:])
        if not temp_buffer.empty:
            temp_buffer.dropna(inplace=True)
            try:
                return np.percentile(temp_buffer.to_list(), num)
            except Exception as e:
                return pd.Series([np.nan])
        else:
            return 0
    else:
        if not temp_buffer.empty:
            return max(temp_buffer)
        else:
            return 0


def count(data: pd.Series):
    return pd.Series(data.count())

# pdsn = pd.Series([1,2,3,np.nan])
# pds1 = pd.Series([1645,1648,1648,np.nan])

def Theory_Days(*data: pd.Series):
    '''
    #Theory_Days(`Theory_start_time`,`Theory_end_time`,`wtg_actual_expiration`)
    '''
    data = list(data)
    data[0] = pd.to_datetime(data[0])
    data[1] = pd.to_datetime(data[1])
    data[2] = pd.to_datetime(data[2])
    if data[1].any() == data[2].any():
        Theory_Days = (data[1] - data[0]).dt.days
    else:
        Theory_Days = (data[1] + pd.Timedelta(days=1) - data[0]).dt.days
    return Theory_Days


def get_hour(df):
    df['day_hour'] = df.index.strftime('%Y-%m-%d-%H')
    trans_mins_df = df.dropna()

    can_work_df = trans_mins_df.groupby(['day_hour']).count()
    can_work_hour_df = can_work_df[can_work_df['windspeed__mean'] == 6]
    workhour = can_work_hour_df.count()['windspeed__mean']

    return workhour
def count_point(*data: pd.Series):
    #分析CoPQ数据得到对应CBB的各服役月数的各指标
    data_1 = {'tempout__mean': data[0],'windspeed__mean': data[1], 'plc_time': data[2]}
    df = pd.DataFrame(data_1)
    df['plc_time'] = pd.to_datetime(df['plc_time'])
    df = df.sort_values(by='plc_time')
    date_max = df['plc_time'].max()
    df = df.set_index('plc_time')
    month_value = date_max.month
    year_value = date_max.year
    if month_value == 6:
        df_front = df.loc[:str(datetime(year_value, month_value, 15, 23))].between_time('7:00', '18:50')
        df_latter = df.loc[str(datetime(year_value, month_value, 16, 0)):]
        df_latter = df_latter.drop(df_latter.between_time('11:00', '16:50').index)
        front_hour = get_hour(df_front)
        latter_hour = get_hour(df_latter) / 2
        workhour = front_hour + latter_hour

    elif month_value in [7, 8]:
        finaldata = df.drop(df.between_time('11:00', '16:50').index)
        workhour = get_hour(finaldata) / 2

    elif month_value == 9:
        df_front = df.loc[:str(datetime(year_value, month_value, 15, 23))]
        df_front = df_front.drop(df_front.between_time('11:00', '16:50').index)
        df_latter = df.loc[str(datetime(year_value, month_value, 16, 0)):].between_time('7:00', '18:50')
        front_hour = get_hour(df_front) / 2
        latter_hour = get_hour(df_latter)
        workhour = front_hour + latter_hour

    else:
        finaldata = df.between_time('7:00', '18:50')
        workhour = get_hour(finaldata)
    return workhour
# # pdsn = pd.Series([2])
# # pds1 = pd.Series([30])
# # r = eql(pds1,pdsn,m=4)
# # print(r )
# v = twr_fatigue_analysis(pds1,pds1,out_type="noDir")
# print(v)
# p = twr_fatigue_analysis(pds1,pdsn,out_type="wDir")
# print(v)
# c =extrapolate(sum(pdsn),sum(pds1),1)
# c1 = pd.Series(c)
# print(c)
# a = gbx_load_abs(pdsn,pds1,method="P99")
# print(a)
# b =sum(pdsn)/sum(pds1)
# print(b)

# d =twr_bias_prctile(pdsn, p=100)
# print(d)
