import pandas as pd
import numpy as np
import random

"""
连续值确定分位数分桶

"""


def generate_percentile_point(src_data, interval):
    '''
    :param src_data: array_like
    :param interval: 百分位数区间, eg:10, 0%,10%,20%....,100%分位数
    :return:
    '''
    bin_res = []
    for i_t in range(0, 101, interval):
        bin_res.append(np.percentile(src_data, i_t))  # 若keepdims=True保持维度不变, sklearn使用fit方便
    bin_res[-1] += 1e5  # 防止使用pd.cut，最大值成为nan
    return bin_res


def get_bucket_result(src_data, bin_point, right):
    '''
    :param src_data: array_like
    :param bin_point: 分隔点:list
    :param right: False 左闭右开区间 eg:[103.0, 196.0)
    :return:
    '''
    bucket_res = pd.cut(src_data, bin_point, right=right)
    return bucket_res


if __name__ == '__main__':

    data_df = pd.DataFrame(columns=['feature'])
    # 随机生成1000个0到999整数
    data_df['feature'] = [random.randint(0, 999) for _range in range(1000)]

    bin_ = generate_percentile_point(data_df['feature'], 10)
    print('分位数点:{}'.format(bin_))

    print('分位数点:{}'.format(np.array(bin_).round(2)))

    # 对随机数进行切分，right=False时左闭右开
    data_df['box'] = get_bucket_result(data_df['feature'], bin_, False)
    stat_ = data_df.groupby('box').agg('count')
    print('================分箱统计结果=====================')
    print(stat_)

    # 为分箱生成对应的标签
    label = []
    for i in range(len(bin_) - 1):
        label.append(str(bin_[i].round(4)) + '+')

    # 原标签和自定义的新标签生成字典
    list_box_td = list(set(data_df['box']))
    list_box_td.sort()
    dict_t = dict(zip(list_box_td, label))

    # 根据字典进行替换

    data_df['label_box'] = data_df['box'].replace(dict_t)
    print('===============分箱标签统计================')
    stat_ = data_df.groupby('label_box').agg('count')
    print(stat_)

    print(data_df.head())