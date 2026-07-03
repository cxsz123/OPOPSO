import torch

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局参数
dim = 30
MaxFEs = 10000 * dim
initial_pop_size = 1000  # 初始种群规模
target_pop_size = int(0.1*initial_pop_size)  # 目标种群规模
FunRunNum = [i for i in range(1, 31) if i != 2]
funNum = len(FunRunNum)
counternum = 50  # 等分FEs记录点数量
dim_list = [30, 50, 100]   # 循环测试维度列表