import numpy as np


def calculate_mean_variance(data):
    """
    计算一组数据的均值和方差。

    参数:
    data (list or numpy.ndarray): 数据列表或数组。

    返回:
    mean (float): 数据的均值。
    variance (float): 数据的方差。
    """
    # 将数据转换为 NumPy 数组
    data_array = np.array(data)

    # 计算均值
    mean = np.mean(data_array)

    # 计算方差
    variance = np.var(data_array)

    return mean, variance


# 示例数据
data = [77, 76, 78, 73, 77]

# 计算均值和方差
mean, variance = calculate_mean_variance(data)

# 输出结果
print(f"均值: {mean}")
print(f"方差: {variance}")
