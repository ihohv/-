import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 设定允许的误差精度
E = 0.03

# 设定信度水平对应的alpha值
alpha95 = 0.05
alpha90 = 0.10

# 计算对应信度下的z值
Z95 = norm.ppf(1 - alpha95)  # 对于95%信度
Z90 = norm.ppf(1 - alpha90)  # 对于90%信度

p0 = 0.1


# 定义计算样本量的函数
def calc_n_size(z, p0, E):
    return (z * np.sqrt(p0 * (1 - p0)) / E) ** 2


def calc_size_and_threshold(size, p, level, accept=True):
    n = size
    while True:
        if accept:
            cum_prob = binom.cdf(np.arange(n + 1), n, p)
            value = np.where(cum_prob >= level)[0][0] - 1
        else:
            cum_prob = 1 - binom.cdf(np.arange(n + 1), n, p)
            if np.any(cum_prob <= 1 - level):
                value = np.where(cum_prob <= 1 - level)[0][0] - 1
            else:
                value = n
        if (accept and cum_prob[value + 1] >= level) or \
                (not accept and cum_prob[value + 1] <= 1 - level):
            break
        n += 10
    return n, value


n_95 = np.ceil(calc_n_size(Z95, p0, E))
n_90 = np.ceil(calc_n_size(Z90, p0, E))

print(n_90, n_95)
n_reject, threshold_reject = calc_size_and_threshold(n_95, p0, 0.95, accept=False)
print(f"拒绝情形：样本大小 = {n_reject}，拒绝临界值 = {threshold_reject}")
n_accept, threshold_accept = calc_size_and_threshold(n_90, p0, 0.90, accept=True)
print(f"接受情形：样本大小 = {n_accept}，接受临界值 = {threshold_accept}")

accept_errors = np.linspace(0.01, 0.05, 50)
samples95 = [calc_n_size(Z95, p0, error) for error in accept_errors]
samples90 = [calc_n_size(Z90, p0, error) for error in accept_errors]

plt.figure(figsize=(10, 6))
plt.plot(accept_errors, samples95, label='95% Confidence', color='red', linestyle='--', linewidth=2)
plt.plot(accept_errors, samples90, label='90% Confidence', color='blue', linestyle='-', linewidth=2)
plt.axvline(x=E, color='g', linestyle='-', linewidth=1)
plt.xlabel('允许的误差 (E)', fontsize=12)
plt.ylabel('样本大小 (n)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
