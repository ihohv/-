import math
from random import randint, random
import matplotlib.pyplot as plt

# 定义成本参数（从表2中获取）
zero_part_costs = [1, 1, 2, 1, 1, 2, 1, 2]  # 零配件检测成本
half_product_costs = [8, 8, 8]  # 半成品装配成本
product_costs = 8  # 成品装配成本
product_check_cost = 6  # 成品检测成本
swap_loss = 40  # 调换损失
rework_cost = 10  # 拆解费用

# 零配件和半成品的次品率（从表中获取）
zero_part_defect_rates = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
half_product_defect_rates = [0.10, 0.10, 0.10]
product_defect_rate = 0.10


# 成本函数定义
def calc_total_cost(detect_zero_parts, detect_half_products, detect_final_product, rework):
    # 零配件检测成本
    zero_part_cost = sum(zero_part_costs[i] if detect_zero_parts[i] else 0 for i in range(8))

    # 半成品装配成本
    half_product_cost = sum(half_product_costs)

    # 半成品检测成本
    half_product_check_cost = sum([half_product_costs[i] if detect_half_products[i] else 0 for i in range(3)])

    # 成品检测成本
    product_check_total_cost = product_check_cost if detect_final_product else 0

    # 不合格产品处理成本
    if rework:
        product_defect_total_cost = product_defect_rate * (swap_loss + rework_cost)
    else:
        product_defect_total_cost = product_defect_rate * swap_loss

    # 总成本计算
    total_cost = zero_part_cost + half_product_cost + half_product_check_cost + product_check_total_cost + product_defect_total_cost
    return total_cost


class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = calc_total_cost
        self.iter = iter  # 内循环迭代次数,即为L =100
        self.alpha = alpha  # 降温系数，alpha=0.99
        self.T0 = T0  # 初始温度T0为100
        self.Tf = Tf  # 温度终值Tf为0.01
        self.T = T0  # 当前温度
        self.detect_zero_parts = [[randint(0, 1) for _ in range(8)] for _ in range(iter)]  # 随机生成100个x的值
        self.detect_half_products = [[randint(0, 1) for _ in range(3)] for _ in range(iter)]  # 随机生成100个y的值
        self.detect_final_product = [randint(0, 1) for _ in range(iter)]
        self.rework = [randint(0, 1) for _ in range(iter)]
        self.most_best = []
        self.history = {'f': [], 'T': []}

    def generate_new(self):  # 扰动产生新解的过程
        detect_zero_parts_new = [randint(0, 1) for _ in range(8)]
        detect_half_products = [randint(0, 1) for _ in range(3)]
        detect_final_product_new = randint(0, 1)
        rework_new = randint(0, 1)
        return detect_zero_parts_new, detect_half_products, detect_final_product_new, rework_new

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):  # 获取最优目标函数值
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.detect_zero_parts[i], self.detect_half_products[i], self.detect_final_product[i],
                          self.rework[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:

            # 内循环迭代100次
            for i in range(self.iter):
                f = self.func(self.detect_zero_parts[i], self.detect_half_products[i], self.detect_final_product,
                              self.rework)  # f为迭代一次后的值
                detect_zero_parts_new, detect_half_products, detect_final_product_new, rework_new = self.generate_new()  # 产生新解
                f_new = self.func(detect_zero_parts_new, detect_half_products, detect_final_product_new,
                                  rework_new)  # 产生新值
                if self.Metrospolis(f, f_new):  # 判断是否接受新值
                    self.detect_zero_parts[i] = detect_zero_parts_new  # 如果接受新值，则把新值的x,y存入x数组和y数组
                    self.detect_half_products[i] = detect_half_products
                    self.detect_final_product[i] = detect_final_product_new
                    self.rework[i] = rework_new
            # 迭代L次记录在该温度下最优解
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            count += 1

            # 得到最优解
        f_best, idx = self.best()
        print(f"F={f_best}, detect_zero_parts={self.detect_zero_parts[idx]}, "
              f"detect_half_products={self.detect_half_products[idx]},"
              f"detect_final_product={self.detect_final_product[idx]},"
              f"rework={self.rework[idx]}")


sa = SA(calc_total_cost)
sa.run()

plt.plot(sa.history['T'], sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()
